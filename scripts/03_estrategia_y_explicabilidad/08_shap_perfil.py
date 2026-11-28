"""08 — SHAP de un perfil (cascada).

Objetivo:
    Explicar la tasa de S3 de un perfil: gráfica de cascada con Plotly (go.Waterfall) desde
    el valor base hasta la salida del modelo, tabla de contribuciones por variable y las
    frases de lectura (tres que suben, tres que bajan, nota de no causalidad).

    Explicar también la línea de S1 del mismo perfil con su descomposición exacta:
    promedio general × factor del producto × factor del score (cascada en log, con los
    factores como etiqueta), tabla y frases. S1 sí se explica para DARIA 2.0 (tiene MOB 0);
    S3 no, porque no es evaluable en riesgo.

    El perfil se da por parámetros. Si no se da ninguno, se usa el perfil de ejemplo: la
    cohorte de prueba con más cuentas expuestas en la vintage de prueba más reciente.

Entradas:
    - models/s3/ y models/s1/ (modelos publicados).
    - data/processed/s3_modelado.csv (solo para elegir el perfil de ejemplo).

Salidas:
    - reports/03_estrategia_y_explicabilidad/shap_s3_perfil_contribuciones.csv
    - reports/03_estrategia_y_explicabilidad/shap_s3_perfil_frases.txt
    - reports/03_estrategia_y_explicabilidad/graficas/shap_s3_perfil_cascada.html
    - reports/03_estrategia_y_explicabilidad/shap_s1_perfil_contribuciones.csv
    - reports/03_estrategia_y_explicabilidad/shap_s1_perfil_frases.txt
    - reports/03_estrategia_y_explicabilidad/graficas/shap_s1_perfil_cascada.html

Cómo correrlo (desde la raíz del repo):
    python scripts/03_estrategia_y_explicabilidad/08_shap_perfil.py
    python scripts/03_estrategia_y_explicabilidad/08_shap_perfil.py --estrategia "r.Reca champ" --segmento Costco \\
        --producto Costco --canal Costco --edad "02| Young 25-30" --score 141-150 --vintage 202601

Listo cuando:
    Un perfil de ejemplo produce su cascada y su tabla de contribuciones, para S1 y S3.
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import catalog as cat  # noqa: E402
from src.explain import s1_factors  # noqa: E402
from src.explain.phrases import as_text, causal_terms, reading_phrases  # noqa: E402
from src.explain.plots import waterfall  # noqa: E402
from src.explain.shap_values import VARIABLE_LABELS, VARIABLES, additivity_error, shap_by_variable, variable_values  # noqa: E402
from src.features.engineering import TEST, build_features  # noqa: E402
from src.models.predict import load_s1, load_s3, platt_transform, sigmoid  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402
from src.utils.charts import save_html  # noqa: E402

ARGS_TO_COLUMNS = {"estrategia": "STRATEGY", "segmento": "STRATEGY_SEGMENT", "producto": "PRODUCT", "canal": "CHANNEL",
                   "edad": "AGE_TAG", "score": "SCORE_TAG", "vintage": "VINTAGE"}


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Cascada SHAP de S3 para un perfil, con su tabla de contribuciones y frases de lectura. "
                    "Sin parámetros usa el perfil de ejemplo; si se da alguno, hay que darlos todos."
    )
    for name, col in ARGS_TO_COLUMNS.items():
        parser.add_argument(f"--{name}", default=None, help=f"Valor de {col}" + (" (AAAAMM; de ella sale el mes de booking)" if col == "VINTAGE" else ""))
    parser.add_argument("--sin-graficas", action="store_true", help="No genera la gráfica HTML.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def example_profile(config: dict) -> pd.DataFrame:
    """Cohorte de prueba con más cuentas expuestas en la vintage de prueba más reciente."""
    frame = pd.read_csv(get_path(config, "processed_dir") / "s3_modelado.csv", encoding="utf-8")
    test = frame[frame["CONJUNTO"] == TEST]
    last = test[test["VINTAGE"] == test["VINTAGE"].max()]
    return last.sort_values(["ENRACCTS"] + cat.COHORT_COLUMNS, ascending=[False] + [True] * 7).head(1)[cat.COHORT_COLUMNS].reset_index(drop=True)


def profile_from_args(args: argparse.Namespace) -> pd.DataFrame:
    """Perfil a partir de los parámetros (todos obligatorios si se da alguno)."""
    given = {col: getattr(args, name) for name, col in ARGS_TO_COLUMNS.items()}
    missing = [f"--{n}" for n, c in ARGS_TO_COLUMNS.items() if given[c] is None]
    if missing:
        raise ValueError(f"Faltan parámetros del perfil: {', '.join(missing)}.")
    given["VINTAGE"] = int(given["VINTAGE"])
    return pd.DataFrame([given])[cat.COHORT_COLUMNS]


def compute_s1(config: dict, profile: pd.DataFrame) -> dict:
    """Descomposición exacta de la línea de S1 del perfil, tabla y frases."""
    s1 = load_s1(get_path(config, "models_dir"))
    build_features(profile)  # valida el perfil contra el catálogo
    dec = s1_factors.decompose(s1.model, profile)
    row = dec.iloc[0]
    values = variable_values(profile).iloc[0]
    contrib = row[s1_factors.S1_VARIABLES]
    table = pd.DataFrame({
        "variable": s1_factors.S1_VARIABLES, "etiqueta": [VARIABLE_LABELS[v] for v in s1_factors.S1_VARIABLES],
        "valor": [values[v] for v in s1_factors.S1_VARIABLES], "contribucion_log": contrib.to_numpy(),
        "factor": np.exp(contrib.to_numpy()),
    })
    return {"table": table, "contrib": contrib, "values": values, "base": float(row["VALOR_BASE"]), "log": float(row[s1_factors.SALIDA_LOG]),
            "line": float(np.exp(row[s1_factors.SALIDA_LOG])), "global": float(np.exp(row["VALOR_BASE"])),
            "phrases": s1_factors.reading_phrases(row[VARIABLES], values), "error": s1_factors.additivity_error(dec),
            "level": int(row["NIVEL"]), "version": s1.metadata["version"]}


def compute(config: dict, profile: pd.DataFrame) -> dict:
    """SHAP del perfil, tasas sin calibrar y calibrada, y frases. S3 es None si no es evaluable en riesgo."""
    s1 = compute_s1(config, profile)
    if profile["STRATEGY"].iloc[0] in cat.NOT_RISK_EVALUABLE:
        return {"profile": profile, "s1": s1, "s3": None}
    s3 = load_s3(get_path(config, "models_dir"))
    features = build_features(profile)  # valida el perfil contra el catálogo
    grouped = shap_by_variable(s3.booster, features)
    row = grouped.iloc[0]
    values = variable_values(profile).iloc[0]
    raw = float(sigmoid(row["SALIDA_LOGIT"]))
    calibrated = float(platt_transform(raw, s3.a, s3.b))
    contrib = row[VARIABLES]
    table = pd.DataFrame({
        "variable": VARIABLES, "etiqueta": [VARIABLE_LABELS[v] for v in VARIABLES], "valor": [values[v] for v in VARIABLES],
        "contribucion_logit": [contrib[v] for v in VARIABLES],
    })
    table["direccion"] = table["contribucion_logit"].map(lambda c: "sube" if c > 1e-6 else ("baja" if c < -1e-6 else "neutra"))
    table = table.reindex(table["contribucion_logit"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    phrases = reading_phrases(contrib, values)
    return {"profile": profile, "s1": s1, "s3": {
        "table": table, "contrib": contrib, "values": values, "base": float(row["VALOR_BASE"]),
        "logit": float(row["SALIDA_LOGIT"]), "raw": raw, "calibrated": calibrated, "phrases": phrases,
        "error": additivity_error(grouped), "version": s3.metadata["version"],
    }}


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda las tablas de contribuciones y las frases de S1 y (si es evaluable) de S3."""
    reports = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad"
    reports.mkdir(parents=True, exist_ok=True)
    p = results["profile"].iloc[0]
    header = "Perfil: " + " · ".join(f"{c}={p[c]}" for c in cat.COHORT_COLUMNS)
    written = []
    s1 = results["s1"]
    s1_table, s1_text = reports / "shap_s1_perfil_contribuciones.csv", reports / "shap_s1_perfil_frases.txt"
    s1["table"].to_csv(s1_table, index=False, encoding="utf-8")
    line = f"Línea promedio estimada: {s1['line']:,.0f} (promedio general {s1['global']:,.0f})"
    s1_text.write_text(f"{header}\n{line}\n\n{s1_factors.as_text(s1['phrases'])}\n", encoding="utf-8")
    written += [s1_table, s1_text]
    r = results["s3"]
    if r is not None:
        table_path = reports / "shap_s3_perfil_contribuciones.csv"
        r["table"].to_csv(table_path, index=False, encoding="utf-8")
        rates = f"Tasa estimada a 6 meses: {r['calibrated']:.2%} (sin calibrar {r['raw']:.2%})"
        text_path = reports / "shap_s3_perfil_frases.txt"
        text_path.write_text(f"{header}\n{rates}\n\n{as_text(r['phrases'])}\n", encoding="utf-8")
        written += [table_path, text_path]
    return written


# ===== 4. Gráficas =====
def make_charts(results: dict, config: dict) -> list:
    """Cascadas del perfil: S1 (factores, en log) y S3 (SHAP, en logit) si es evaluable."""
    out = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / "graficas"
    p = results["profile"].iloc[0]
    s1 = results["s1"]
    order = s1["contrib"].abs().sort_values(ascending=False).index
    texts = [f"{s1['global']:,.0f}"] + [f"×{np.exp(s1['contrib'][v]):.3f}" for v in order] + [f"{s1['line']:,.0f}"]
    fig1 = waterfall(s1["contrib"], s1["base"], s1["log"], s1["values"],
                     f"S1: explicación del perfil ({p['PRODUCT']}, {p['SCORE_TAG']}) — línea estimada {s1['line']:,.0f}",
                     y_label="log de la línea promedio", scale_note=s1_factors.S1_SCALE_NOTE,
                     base_label="Promedio general", output_label="Línea estimada", texts=texts)
    # Eje acotado al recorrido base → salida: en log la base (≈ 10.8) aplastaría los factores.
    path = np.cumsum([s1["base"]] + [s1["contrib"][v] for v in order])
    span = max(path.max() - path.min(), 0.05)
    fig1.update_yaxes(range=[path.min() - 0.6 * span, path.max() + 0.6 * span])
    written = [save_html(fig1, out / "shap_s1_perfil_cascada.html")]
    r = results["s3"]
    if r is not None:
        title = f"S3: explicación del perfil ({p['PRODUCT']}, {p['CHANNEL']}, {p['SCORE_TAG']}) — tasa estimada {r['calibrated']:.2%}"
        written.append(save_html(waterfall(r["contrib"], r["base"], r["logit"], r["values"], title), out / "shap_s3_perfil_cascada.html"))
    return written


# ===== 5. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si el perfil no es válido."""
    args = parse_args()
    config = load_config()
    set_seed(config["seed"])
    try:
        given = any(getattr(args, n) is not None for n in ARGS_TO_COLUMNS)
        profile = profile_from_args(args) if given else example_profile(config)
        results = compute(config, profile)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    written = save_results(results, config)
    if not args.sin_graficas:
        written += make_charts(results, config)

    p = results["profile"].iloc[0]
    print(f"Perfil{' de ejemplo' if not given else ''}: " + " · ".join(f"{p[c]}" for c in cat.COHORT_COLUMNS))
    s1 = results["s1"]
    factors = " × ".join(f"{np.exp(s1['contrib'][v]):.3f}" for v in s1_factors.S1_VARIABLES)
    print(f"S1 ({s1['version']}): promedio general {s1['global']:,.0f} × {factors} = línea {s1['line']:,.0f} (error en log {s1['error']:.1e})")
    print(s1_factors.as_text(s1["phrases"]).replace("\n", "\n  ").join(["  ", ""]))
    r = results["s3"]
    if r is None:
        print(f"S3: {p['STRATEGY']} no es evaluable en riesgo (solo tiene MOB 0 y 1); no hay explicación de la tasa.")
    else:
        print(f"S3 ({r['version']}): valor base {r['base']:+.4f} + Σ contribuciones {r['contrib'].sum():+.4f} = logit {r['logit']:+.4f} "
              f"(error {r['error']:.1e})")
        print(f"  Tasa sin calibrar {r['raw']:.4f}; calibrada {r['calibrated']:.4f}")
        print(as_text(r["phrases"]).replace("\n", "\n  ").join(["  ", ""]))
    texts = s1_factors.as_text(s1["phrases"]) + ("" if r is None else as_text(r["phrases"]))
    print(f"  Lenguaje causal en las frases: {causal_terms(texts) or 'ninguno'}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
