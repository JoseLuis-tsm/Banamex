"""09 — Frases de lectura de SHAP.

Objetivo:
    Generar por plantilla (sin modelo de lenguaje) las frases de lectura de cada cohorte de
    prueba de S3: los tres factores que más suben y los tres que más bajan la tasa estimada,
    y la nota fija de no causalidad. Lo mismo para la línea de S1 en sus cohortes de prueba
    (cada factor distinto de 1, con su valor). Verificar que ninguna frase use lenguaje causal
    y que toda gráfica SHAP generada (de S1 y de S3) lleve la nota.

Entradas:
    - data/processed/shap_s3.csv y shap_s1.csv (de 06_shap_global.py).
    - reports/03_estrategia_y_explicabilidad/graficas/shap_*.html (de 06, 07 y 08).
    - src/explain/phrases.py y src/explain/s1_factors.py.

Salidas:
    - reports/03_estrategia_y_explicabilidad/shap_s3_frases_prueba.csv: por cohorte de
      prueba, sus dimensiones y las frases (suben, bajan, nota).
    - reports/03_estrategia_y_explicabilidad/shap_s1_frases_prueba.csv: igual, para S1.
    - reports/03_estrategia_y_explicabilidad/shap_verificacion_nota.csv: por gráfica SHAP,
      si lleva la nota.

Cómo correrlo (desde la raíz del repo):
    python scripts/03_estrategia_y_explicabilidad/09_frases_lectura.py

Listo cuando:
    Toda vista con SHAP lleva la nota y las frases no usan lenguaje causal (si no, sale con
    código 1).
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402
import json  # noqa: E402

import pandas as pd  # noqa: E402

from src.data import catalog as cat  # noqa: E402
from src.explain import s1_factors  # noqa: E402
from src.explain.phrases import as_text, causal_terms, reading_phrases  # noqa: E402
from src.explain.shap_values import NOTE, SHAP_PREFIX, VARIABLES, from_table, variable_values  # noqa: E402
from src.features.engineering import TEST  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Genera las frases de lectura SHAP de las cohortes de prueba y verifica la nota y la ausencia de lenguaje causal."
    )
    parser.add_argument("--sin-graficas", action="store_true", help="Este paso no genera gráficas; se acepta por uniformidad.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def phrases_for(table: pd.DataFrame, contrib: pd.DataFrame, make, to_text):
    """Frases de cada fila con make(contribuciones, valores) y cuántas usan lenguaje causal."""
    values = variable_values(table)
    rows, causal = [], 0
    for i in range(len(table)):
        p = make(contrib.loc[i, VARIABLES], values.loc[i])
        found = causal_terms(to_text(p))
        causal += bool(found)
        rows.append({**{c: table.loc[i, c] for c in cat.COHORT_COLUMNS},
                     "frases_suben": json.dumps(p["suben"], ensure_ascii=False), "frases_bajan": json.dumps(p["bajan"], ensure_ascii=False),
                     "nota": p["nota"], "terminos_causales": "; ".join(found)})
    return pd.DataFrame(rows), causal


def compute(config: dict) -> dict:
    """Frases por cohorte de prueba (S3 y S1) y verificación de la nota en las gráficas SHAP."""
    processed = get_path(config, "processed_dir")
    table = pd.read_csv(processed / "shap_s3.csv", encoding="utf-8")
    test = table[table["CONJUNTO"] == TEST].reset_index(drop=True)
    phrases, causal = phrases_for(test, from_table(test), reading_phrases, as_text)
    t1 = pd.read_csv(processed / "shap_s1.csv", encoding="utf-8")
    t1 = t1[t1["CONJUNTO"] == TEST].reset_index(drop=True)
    c1 = t1[[SHAP_PREFIX + v for v in VARIABLES]].set_axis(VARIABLES, axis=1)
    phrases_s1, causal_s1 = phrases_for(t1, c1, s1_factors.reading_phrases, s1_factors.as_text)

    # Qué hace: busca la nota en el HTML de cada gráfica SHAP. Por qué: CLAUDE.md pide que
    # toda vista con SHAP la lleve; se verifica sobre los archivos, no sobre el código.
    charts = sorted((get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / "graficas").glob("shap_*.html"))
    # Plotly guarda el texto en JSON con los acentos escapados (ó): se busca en ambas formas.
    escaped = json.dumps(NOTE)[1:-1]
    note_check = pd.DataFrame([{"grafica": c.name, "lleva_nota": any(n in c.read_text(encoding="utf-8") for n in (NOTE, escaped))} for c in charts])
    return {"phrases": phrases, "causal": causal, "phrases_s1": phrases_s1, "causal_s1": causal_s1, "notes": note_check}


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    reports = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad"
    outputs = [(results["phrases"], reports / "shap_s3_frases_prueba.csv"), (results["phrases_s1"], reports / "shap_s1_frases_prueba.csv"),
               (results["notes"], reports / "shap_verificacion_nota.csv")]
    for table, path in outputs:
        table.to_csv(path, index=False, encoding="utf-8")
    return [p for _, p in outputs]


# ===== 4. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si hay lenguaje causal, falta la nota o no hay gráficas."""
    parse_args()
    config = load_config()
    set_seed(config["seed"])
    results = compute(config)
    written = save_results(results, config)

    ph, notes = results["phrases"], results["notes"]
    print(f"Frases de lectura de {len(ph):,} cohortes de prueba; con lenguaje causal: {results['causal']}")
    print("  Ejemplo:")
    first = ph.iloc[0]
    for f in json.loads(first["frases_suben"]) + json.loads(first["frases_bajan"]):
        print(f"    - {f}")
    print(f"    {first['nota']}")
    ph1 = results["phrases_s1"]
    print(f"Frases de lectura de S1 en {len(ph1):,} cohortes de prueba; con lenguaje causal: {results['causal_s1']}")
    first = ph1.iloc[0]
    for f in json.loads(first["frases_suben"]) + json.loads(first["frases_bajan"]):
        print(f"    - {f}")
    print(f"Gráficas SHAP revisadas: {len(notes)}; con la nota: {int(notes['lleva_nota'].sum()) if len(notes) else 0}")
    for _, row in notes[~notes["lleva_nota"]].iterrows() if len(notes) else []:
        print(f"  SIN NOTA: {row['grafica']}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    ok = results["causal"] == 0 and results["causal_s1"] == 0 and len(notes) > 0 and bool(notes["lleva_nota"].all())
    if len(notes) == 0:
        print("FALLA: no hay gráficas SHAP; corre antes 06, 07 y 08 sin --sin-graficas.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
