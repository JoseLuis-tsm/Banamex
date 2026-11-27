"""10 — Verificación de aditividad de SHAP.

Objetivo:
    Comprobar que valor base + suma de contribuciones SHAP = salida del modelo, con
    tolerancia de 0.0001, en los tres niveles: global (todas las cohortes), segmento (cada
    valor de estrategia, producto, canal y edad) y perfil (el perfil de ejemplo y uno por
    estrategia evaluable).

    S3: la comparación es contra la salida **antes de la calibración** (logit del XGBoost).
    SHAP explica el modelo sin calibrar; la calibración de Platt, σ(a·logit + b), es una
    transformación creciente posterior y no se reparte entre las variables. Se reporta
    también la diferencia contra el logit calibrado, para dejarlo medido.
    S1: el publicado es la línea base simple; su explicación es la descomposición exacta
    log(línea) = log promedio global + log factor del producto + log factor del score, y se
    compara contra log(línea predicha por el S1 publicado). La columna de diferencia contra el
    calibrado no aplica (NaN).

Entradas:
    - models/s3/ y models/s1/ (modelos publicados); data/processed/s3_modelado.csv y s1_modelado.csv.

Salidas:
    - reports/03_estrategia_y_explicabilidad/shap_aditividad.csv: por modelo, nivel y grupo,
      filas, error máximo contra la salida sin calibrar, si pasa, y diferencia máxima contra
      el logit calibrado.

Cómo correrlo (desde la raíz del repo):
    python scripts/03_estrategia_y_explicabilidad/10_verificar_aditividad.py

Listo cuando:
    La prueba pasa para cada modelo (si no, sale con código 1).
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402
import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import catalog as cat  # noqa: E402
from src.explain import s1_factors  # noqa: E402
from src.explain.shap_values import VARIABLES, shap_by_variable  # noqa: E402
from src.features.engineering import build_features, feature_columns  # noqa: E402
from src.models.predict import load_s1, load_s3, logit, platt_transform, sigmoid  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402

TOLERANCE = 1e-4
SEGMENT_VARIABLES = ["STRATEGY", "PRODUCT", "CHANNEL", "AGE_TAG"]


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Verifica que valor base + Σ SHAP = salida del modelo (tolerancia 0.0001) en los tres niveles.")
    parser.add_argument("--sin-graficas", action="store_true", help="Este paso no genera gráficas; se acepta por uniformidad.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def check(grouped: pd.DataFrame, calibrated_logit: np.ndarray, model: str, level: str, group: str) -> dict:
    """Error de aditividad contra la salida sin calibrar y diferencia contra la calibrada."""
    total = grouped["VALOR_BASE"] + grouped[VARIABLES].sum(axis=1)
    err = float(np.max(np.abs(total - grouped["SALIDA_LOGIT"])))
    return {"modelo": model, "nivel": level, "grupo": group, "filas": len(grouped), "error_max_sin_calibrar": err,
            "pasa": err <= TOLERANCE, "diferencia_max_vs_calibrado": float(np.max(np.abs(total - calibrated_logit)))}


def example_profiles(config: dict) -> pd.DataFrame:
    """Perfil de ejemplo de 08_shap_perfil.py y una cohorte de prueba por estrategia evaluable."""
    spec = importlib.util.spec_from_file_location("perfil", Path(__file__).with_name("08_shap_perfil.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.read_csv(get_path(config, "processed_dir") / "s3_modelado.csv", encoding="utf-8")
    per_strategy = frame.sort_values(cat.COHORT_COLUMNS).groupby("STRATEGY").head(1)[cat.COHORT_COLUMNS]
    return pd.concat([module.example_profile(config), per_strategy], ignore_index=True)


def compute(config: dict) -> dict:
    """Aditividad de S3 en los tres niveles."""
    s3 = load_s3(get_path(config, "models_dir"))
    frame = pd.read_csv(get_path(config, "processed_dir") / "s3_modelado.csv", encoding="utf-8")
    grouped = shap_by_variable(s3.booster, frame[feature_columns()])
    # Logit de la tasa calibrada: logit(σ(a·z + b)) = a·z + b, con z la salida sin calibrar.
    cal = logit(platt_transform(sigmoid(grouped["SALIDA_LOGIT"]), s3.a, s3.b))
    rows = [check(grouped, cal, "S3", "global", "todas las cohortes")]
    for var in SEGMENT_VARIABLES:
        for value, idx in frame.groupby(var).groups.items():
            rows.append(check(grouped.loc[idx], cal[frame.index.get_indexer(idx)], "S3", "segmento", f"{var} = {value}"))
    profiles = example_profiles(config)
    pg = shap_by_variable(s3.booster, build_features(profiles))
    pcal = logit(platt_transform(sigmoid(pg["SALIDA_LOGIT"]), s3.a, s3.b))
    for i in range(len(profiles)):
        p = profiles.iloc[i]
        label = ("ejemplo: " if i == 0 else "") + " · ".join(str(p[c]) for c in ("STRATEGY", "PRODUCT", "CHANNEL", "SCORE_TAG"))
        rows.append(check(pg.iloc[[i]], pcal[i:i + 1], "S3", "perfil", label))
    rows += compute_s1(config, profiles)
    return {"table": pd.DataFrame(rows), "version": s3.metadata["version"], "version_s1": load_s1(get_path(config, "models_dir")).metadata["version"]}


def check_s1(dec: pd.DataFrame, level: str, group: str) -> dict:
    """Error de log g + Σ contribuciones contra log(línea predicha)."""
    total = dec["VALOR_BASE"] + dec[VARIABLES].sum(axis=1)
    err = float(np.max(np.abs(total - dec[s1_factors.SALIDA_LOG])))
    return {"modelo": "S1", "nivel": level, "grupo": group, "filas": len(dec), "error_max_sin_calibrar": err,
            "pasa": err <= TOLERANCE, "diferencia_max_vs_calibrado": np.nan}


def compute_s1(config: dict, profiles: pd.DataFrame) -> list:
    """Aditividad de la descomposición de S1 en los tres niveles (mismos grupos y perfiles que S3)."""
    s1 = load_s1(get_path(config, "models_dir"))
    frame = pd.read_csv(get_path(config, "processed_dir") / "s1_modelado.csv", encoding="utf-8")
    dec = s1_factors.decompose(s1.model, frame[cat.COHORT_COLUMNS])
    rows = [check_s1(dec, "global", "todas las cohortes de MOB 0")]
    for var in SEGMENT_VARIABLES:
        for value, idx in frame.groupby(var).groups.items():
            rows.append(check_s1(dec.loc[idx], "segmento", f"{var} = {value}"))
    pdec = s1_factors.decompose(s1.model, profiles)
    for i in range(len(profiles)):
        p = profiles.iloc[i]
        label = ("ejemplo: " if i == 0 else "") + " · ".join(str(p[c]) for c in ("STRATEGY", "PRODUCT", "CHANNEL", "SCORE_TAG"))
        rows.append(check_s1(pdec.iloc[[i]], "perfil", label))
    return rows


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    out = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / "shap_aditividad.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results["table"].to_csv(out, index=False, encoding="utf-8")
    return [out]


# ===== 4. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si algún nivel de S1 o de S3 no pasa."""
    parse_args()
    config = load_config()
    set_seed(config["seed"])
    results = compute(config)
    written = save_results(results, config)

    t = results["table"]
    s3 = t[t["modelo"] == "S3"]
    print(f"Aditividad SHAP (tolerancia {TOLERANCE}), S3 modelo {results['version']}, contra la salida sin calibrar:")
    for level in ("global", "segmento", "perfil"):
        s = s3[s3["nivel"] == level]
        print(f"  {level:9s} {len(s):>3} grupos  {int(s['filas'].sum()):>7,} filas  error máximo {s['error_max_sin_calibrar'].max():.1e}  "
              f"{'pasa' if s['pasa'].all() else 'NO PASA'}")
    print(f"  Contra el logit calibrado la diferencia máxima es {s3['diferencia_max_vs_calibrado'].max():.4f}: "
          "SHAP explica el modelo sin calibrar (la calibración es una transformación posterior).")
    s1 = t[t["modelo"] == "S1"]
    print(f"Aditividad de la descomposición de S1 (modelo {results['version_s1']}), contra log(línea predicha):")
    for level in ("global", "segmento", "perfil"):
        s = s1[s1["nivel"] == level]
        print(f"  {level:9s} {len(s):>3} grupos  {int(s['filas'].sum()):>7,} filas  error máximo {s['error_max_sin_calibrar'].max():.1e}  "
              f"{'pasa' if s['pasa'].all() else 'NO PASA'}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0 if bool(t["pasa"].astype(bool).all()) else 1


if __name__ == "__main__":
    sys.exit(main())
