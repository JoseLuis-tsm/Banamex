"""05 — Reglas de consistencia.

Objetivo:
    Aplicar las reglas de src/consistency/rules.py a las filas de prueba y verificar que se
    cumplan en el 100% de ellas:
      - S2 se calcula (cuentas planeadas × S1), no se predice.
      - La tasa de S3 se recorta a [0, 1].
      - DARIA 2.0 sale "no evaluable en riesgo".
      - Un perfil fuera del dominio de entrenamiento de un modelo no recibe su salida.
      - Confianza por cuentas maduras en perfiles similares (Alta ≥ 1,000; Media 200–999;
        Baja < 200).
      - Alerta de revisión cuando el riesgo es Crítico (no aprueba ni rechaza).
      - Cada fila lleva las versiones de los modelos, de los datos y el horizonte.

    Filas de prueba: las cohortes de prueba de S1 (MOB 0, vintages 202512–202609); incluyen
    DARIA 2.0. Las cuentas planeadas de cada fila son sus ACCOUNTS (solo para ejercitar S2).

    Se corre dos veces:
      - producción: el dominio es el de los modelos publicados (todas sus vintages de
        entrenamiento), como lo usará la app.
      - diagnóstico: el dominio es lo que vio el modelo de evaluación antes de la prueba
        (entrenamiento + validación, que usó para hiperparámetros y calibración), para
        ejercitar la regla de "fuera de dominio" con perfiles que ese modelo no vio.

    Umbrales de riesgo: se usan los del metadata.json de S3 (P50, P80 y P95 de la tasa
    predicha en entrenamiento, ponderada por ENRACCTS). La propuesta pedía percentiles de la
    tasa histórica observada; la observada es muy ruidosa por cohorte (35% de las cohortes
    tienen tasa 0), así que se conserva la predicha. Desviación aprobada por Luis y documentada
    en docs/consistencia_fase3.md.

Entradas:
    - models/s1/ y models/s3/ (modelos publicados y metadata.json).
    - data/processed/s1_modelado.csv y s3_modelado.csv.

Salidas:
    - reports/03_estrategia_y_explicabilidad/consistencia_prueba.csv: filas de prueba con
      todas las salidas tras las reglas (corrida de producción).
    - reports/03_estrategia_y_explicabilidad/consistencia_reglas.csv: por corrida y regla,
      filas revisadas, filas que cumplen y si cumple al 100%.
    - reports/03_estrategia_y_explicabilidad/consistencia_resumen.csv: estado, confianza y
      nivel de riesgo por corrida.

Cómo correrlo (desde la raíz del repo):
    python scripts/03_estrategia_y_explicabilidad/05_reglas_consistencia.py

Listo cuando:
    Las reglas se cumplen en el 100% de las filas de prueba en las dos corridas (si no, sale
    con código 1).
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402

from src.consistency.rules import apply_rules, check_rules, domain_keys, in_domain, similar_accounts_table  # noqa: E402
from src.data import catalog as cat  # noqa: E402
from src.features.engineering import TEST  # noqa: E402
from src.models.predict import load_s1, load_s3  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402

RUN_PRODUCTION = "producción"
RUN_DIAGNOSTIC = "diagnóstico"


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Aplica las reglas de consistencia a las filas de prueba y verifica que se cumplan en el 100%% de ellas."
    )
    parser.add_argument("--sin-graficas", action="store_true", help="Este paso no genera gráficas; se acepta por uniformidad.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def compute(config: dict) -> dict:
    """Predice las filas de prueba, aplica las reglas en las dos corridas y las verifica."""
    processed, models = get_path(config, "processed_dir"), get_path(config, "models_dir")
    s1_frame = pd.read_csv(processed / "s1_modelado.csv", encoding="utf-8")
    s3_frame = pd.read_csv(processed / "s3_modelado.csv", encoding="utf-8")
    s1, s3 = load_s1(models), load_s3(models)

    test = s1_frame[s1_frame["CONJUNTO"] == TEST].reset_index(drop=True)
    profiles = test[cat.COHORT_COLUMNS]
    line = s1.predict_line(profiles)
    rate = s3.predict_rate(profiles)
    thresholds = {k: s3.metadata["umbrales_riesgo"][k] for k in ("p50", "p80", "p95")}
    versions = {"modelo_s1": s1.metadata["version"], "modelo_s3": s3.metadata["version"],
                "datos_sha256": s3.metadata["datos_sha256"], "horizonte_mob": int(config["horizon_mob"])}
    # Perfiles similares: cohortes maduras a 6 meses (todas las de s3_modelado).
    similar = similar_accounts_table(s3_frame)

    def in_final(frame: pd.DataFrame, meta: dict) -> pd.DataFrame:
        return frame[frame["VINTAGE"].isin(meta["vintages"]["modelo_final"])]

    runs = {
        RUN_PRODUCTION: (domain_keys(in_final(s1_frame, s1.metadata)), domain_keys(in_final(s3_frame, s3.metadata))),
        RUN_DIAGNOSTIC: (domain_keys(s1_frame[s1_frame["CONJUNTO"] != TEST]), domain_keys(s3_frame[s3_frame["CONJUNTO"] != TEST])),
    }
    outputs, checks, summary = {}, [], []
    for name, (d1, d3) in runs.items():
        out = apply_rules(profiles, line, rate, thresholds, d1, d3, similar, versions, planned_accounts=test["ACCOUNTS"].to_numpy())
        outputs[name] = out
        checks.append(check_rules(out, rate, line, thresholds).assign(corrida=name))
        for col in ("estado", "confianza", "nivel_riesgo"):
            c = out[col].fillna("(sin nivel)").value_counts().rename_axis("valor").reset_index(name="filas")
            summary.append(c.assign(corrida=name, variable=col))
    checks = pd.concat(checks, ignore_index=True)[["corrida", "regla", "filas_revisadas", "filas_que_cumplen", "cumple_100"]]
    summary = pd.concat(summary, ignore_index=True)[["corrida", "variable", "valor", "filas"]]
    # Referencia de la propuesta: filas de prueba de S3 fuera del dominio de entrenamiento.
    s3_test = s3_frame[s3_frame["CONJUNTO"] == TEST]
    s3_out_share = 1 - float(in_domain(s3_test, runs[RUN_DIAGNOSTIC][1]).mean())
    return {"out": outputs[RUN_PRODUCTION], "checks": checks, "summary": summary, "n": len(test),
            "thresholds": thresholds, "s3_test_out_share": s3_out_share, "s3_test_rows": len(s3_test)}


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda las filas tras las reglas, la verificación por regla y el resumen."""
    reports = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad"
    reports.mkdir(parents=True, exist_ok=True)
    outputs = [(results["out"], reports / "consistencia_prueba.csv"), (results["checks"], reports / "consistencia_reglas.csv"),
               (results["summary"], reports / "consistencia_resumen.csv")]
    for table, path in outputs:
        table.to_csv(path, index=False, encoding="utf-8")
    return [p for _, p in outputs]


# ===== 4. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si alguna regla no se cumple en el 100% de las filas."""
    parse_args()
    config = load_config()
    set_seed(config["seed"])
    results = compute(config)
    written = save_results(results, config)

    t = results["thresholds"]
    print(f"Reglas de consistencia sobre {results['n']:,} filas de prueba (cohortes de prueba de S1, MOB 0)")
    print(f"  Umbrales de riesgo (metadata de S3): P50 {t['p50']:.4f}, P80 {t['p80']:.4f}, P95 {t['p95']:.4f}")
    s = results["summary"]
    for run in (RUN_PRODUCTION, RUN_DIAGNOSTIC):
        print(f"  Corrida {run}:")
        for var in ("estado", "confianza", "nivel_riesgo"):
            part = s[(s["corrida"] == run) & (s["variable"] == var)]
            print(f"    {var:13s} " + "; ".join(f"{r['valor']} {r['filas']:,}" for _, r in part.iterrows()))
    print(f"  Referencia: filas de prueba de S3 fuera del dominio de entrenamiento (diagnóstico): "
          f"{results['s3_test_out_share']:.1%} de {results['s3_test_rows']:,}")
    c = results["checks"]
    print("Reglas:")
    for _, r in c.iterrows():
        print(f"  [{r['corrida']:11s}] {r['regla']:45s} {r['filas_que_cumplen']:>6,}/{r['filas_revisadas']:<6,} "
              f"{'ok' if r['cumple_100'] else 'FALLA'}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0 if bool(c["cumple_100"].all()) else 1


if __name__ == "__main__":
    sys.exit(main())
