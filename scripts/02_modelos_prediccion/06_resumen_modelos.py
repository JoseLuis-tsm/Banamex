"""06 — Resumen de modelos y fichas de S1 y S3.

Objetivo:
    Juntar en una tabla los resultados de S1 y S3 (modelo contra líneas base, validación
    contra prueba) y escribir una ficha corta de cada modelo en docs/fichas/, con todas las
    cifras tomadas de metadata.json y de los reportes de métricas.

Entradas:
    - models/<salida>/actual.json y metadata.json; si una salida no tiene versión publicada,
      la versión guardada más reciente (se indica en la ficha).
    - reports/02_modelos_prediccion/s1_comparacion.csv y s3_comparacion.csv.

Salidas:
    - reports/02_modelos_prediccion/resumen_modelos.csv
    - docs/fichas/ficha_s1.md y docs/fichas/ficha_s3.md

Cómo correrlo (desde la raíz del repo):
    python scripts/02_modelos_prediccion/06_resumen_modelos.py

Listo cuando:
    Las fichas están completas (las revisa Luis).
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402
from typing import Optional, Tuple  # noqa: E402

import pandas as pd  # noqa: E402

from src.features.engineering import TEST  # noqa: E402
from src.models import registry  # noqa: E402
from src.utils import PROJECT_ROOT, get_path, load_config, set_seed  # noqa: E402


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Genera la tabla resumen de S1 y S3 y las fichas en docs/fichas/.")
    parser.add_argument("--sin-graficas", action="store_true", help="Este paso no genera gráficas; se acepta por uniformidad.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def find_metadata(models_dir: Path, output: str) -> Tuple[Optional[dict], bool]:
    """Metadata de la versión publicada; si no hay, la más reciente guardada. Devuelve (metadata, publicada)."""
    found = registry.current(models_dir, output)
    if found:
        return found[1], True
    candidates = [registry.read_json(p) for p in (Path(models_dir) / output).glob("*/metadata.json")]
    if not candidates:
        return None, False
    return max(candidates, key=lambda m: m["fecha"]), False


def summary_rows(comparison: pd.DataFrame, output: str, metrics: list) -> pd.DataFrame:
    """Filas totales (validación y prueba) de cada modelo con las métricas elegidas."""
    t = comparison[comparison["desglose"] == "Total"][["modelo", "conjunto"] + metrics].copy()
    t.insert(0, "salida", output)
    return t


def compute(config: dict) -> dict:
    """Lee metadata y comparaciones y arma el resumen."""
    reports = get_path(config, "reports_dir") / "02_modelos_prediccion"
    models_dir = get_path(config, "models_dir")
    s1_cmp = pd.read_csv(reports / "s1_comparacion.csv", encoding="utf-8")
    s3_cmp = pd.read_csv(reports / "s3_comparacion.csv", encoding="utf-8")
    summary = pd.concat([
        summary_rows(s1_cmp, "S1", ["mae", "mape_pct", "r2", "sesgo_pct"]),
        summary_rows(s3_cmp, "S3", ["auc", "ks", "gini", "log_loss", "brier", "obs_pred"]),
    ], ignore_index=True)
    s1_meta, s1_pub = find_metadata(models_dir, "s1")
    s3_meta, s3_pub = find_metadata(models_dir, "s3")
    return {"summary": summary, "s1": (s1_meta, s1_pub, s1_cmp), "s3": (s3_meta, s3_pub, s3_cmp)}


# ===== 3. Fichas =====
def md_table(df: pd.DataFrame, formats: dict) -> str:
    """Tabla markdown con formato por columna."""
    header = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    lines = []
    for _, r in df.iterrows():
        # La barra vertical de valores como "01| Young < 25" se escapa para no romper la tabla.
        lines.append("| " + " | ".join(formats.get(c, "{}").format(r[c]).replace("|", r"\|") for c in df.columns) + " |")
    return header + "\n".join(lines)


def common_header(meta: dict, published: bool, title: str) -> str:
    v = meta["vintages"]
    status = "**Publicado** (models/{s}/actual.json)".format(s=meta["salida"]) if published else "**No publicado**: no cumplió el criterio de publicación"
    return (
        f"# Ficha del modelo {title}\n\n"
        "> Generada por `scripts/02_modelos_prediccion/06_resumen_modelos.py` a partir de `metadata.json` y de los reportes de métricas. "
        "No editar a mano: se regenera al correr el script.\n\n"
        f"- **Estado:** {status}.\n"
        f"- **Versión:** `{meta['version']}` · **Datos (SHA-256):** `{meta['datos_sha256'][:16]}…`\n"
        "- **Base de datos sintética:** sí. Las cifras describen la base sintética, no datos reales.\n"
        f"- **Partición temporal por vintage:** entrenamiento {v['entrenamiento'][0]}–{v['entrenamiento'][-1]} ({len(v['entrenamiento'])}), "
        f"validación {v['validacion'][0]}–{v['validacion'][-1]} ({len(v['validacion'])}), prueba {v['prueba'][0]}–{v['prueba'][-1]} ({len(v['prueba'])}). "
        f"El modelo final se reentrena con {len(v['modelo_final'])} vintages.\n"
        "- **Uso de la prueba:** el conjunto de prueba se evaluó **una sola vez**, al final, con el modelo entrenado solo con entrenamiento; "
        "hiperparámetros, retransformación y calibración se eligieron en validación.\n"
        f"- **Variables:** {len(meta['variables'])} (one-hot de estrategia, segmento, producto y canal; edad ordinal; posición del score 0–1 en su familia; familia; mes de booking). "
        "No entran la vintage, el MOB, los pesos ni las columnas de resultado.\n"
        f"- **Hiperparámetros:** profundidad {meta['hiperparametros']['max_depth']}, eta {meta['hiperparametros']['eta']:.4f}, "
        f"{meta['rondas']} rondas; Optuna {meta['busqueda']['pruebas']} pruebas, parada temprana {meta['busqueda']['parada_temprana']}, semilla {meta['semilla']}.\n\n"
    )


def ficha_s1(meta: dict, published: bool, cmp: pd.DataFrame) -> str:
    totals = cmp[cmp["desglose"] == "Total"][["modelo", "conjunto", "mae", "mape_pct", "r2", "sesgo_pct"]]
    by = cmp[(cmp["conjunto"] == TEST) & (cmp["desglose"] == "STRATEGY")][["modelo", "grupo", "mae", "mape_pct", "sesgo_pct", "peso"]]
    fmt = {"mae": "{:,.0f}", "mape_pct": "{:.1f}%", "r2": "{:.3f}", "sesgo_pct": "{:+.1f}%", "peso": "{:,.0f}"}
    c = meta["criterio"]
    r = meta["retransformacion"]
    return (
        common_header(meta, published, "S1 — Línea promedio esperada")
        + "## Qué estima\n\nLa línea de crédito promedio por cuenta que recibe un perfil al booking: "
        "`CREDITLINESAMTS / ACCOUNTS` en MOB = 0. XGBoost sobre el logaritmo de la línea, con peso = `ACCOUNTS`. "
        f"Retransformación elegida en validación: **{r['metodo']}** (factor del modelo final {r['factor']:.4f}).\n\n"
        "## Resultados (totales)\n\n" + md_table(totals, fmt) + "\n\n"
        "## Prueba por estrategia\n\n" + md_table(by, fmt) + "\n\n"
        "## Criterio de publicación\n\n"
        f"{c['regla']}. Mejora obtenida: **{100 * c['mejora']:.1f}%** → **{'aprobado' if c['aprobado'] else 'no aprobado'}**.\n\n"
        "## Limitaciones\n\n"
        "- DARIA solo aparece en validación y prueba: sus métricas miden extrapolación (el modelo no vio DARIA al entrenar).\n"
        "- Ningún modelo explica la variación de la línea entre cohortes de un mismo perfil; el error relativo (MAPE) es parecido en todos los modelos.\n"
        "- La herramienta no aprueba ni rechaza créditos ni predice por cliente individual.\n"
    )


def ficha_s3(meta: dict, published: bool, cmp: pd.DataFrame) -> str:
    keep = cmp["modelo"] != "XGBoost sin calibrar"
    totals = cmp[keep & (cmp["desglose"] == "Total")][["modelo", "conjunto", "auc", "ks", "gini", "log_loss", "brier", "obs_pred"]]
    fmt = {k: "{:.4f}" for k in ("auc", "ks", "gini", "brier")} | {"log_loss": "{:.5f}", "obs_pred": "{:.3f}", "peso": "{:,.0f}"}
    model = cmp["modelo"] == "XGBoost calibrado"
    by_s = cmp[model & (cmp["conjunto"] == TEST) & (cmp["desglose"] == "STRATEGY")][["grupo", "auc", "log_loss", "obs_pred", "peso"]]
    by_a = cmp[model & (cmp["conjunto"] == TEST) & (cmp["desglose"] == "AGE_TAG")][["grupo", "auc", "log_loss", "obs_pred", "peso"]]
    levels = pd.DataFrame(meta["metricas"]["prueba_por_nivel_riesgo"])[["nivel", "cohortes", "cuentas_expuestas", "tasa_observada", "tasa_predicha"]]
    lvl_fmt = {"cohortes": "{:,.0f}", "cuentas_expuestas": "{:,.0f}", "tasa_observada": "{:.2%}", "tasa_predicha": "{:.2%}"}
    cal, u, c = meta["calibracion"], meta["umbrales_riesgo"], meta["criterio"]
    checks = "\n".join(f"- {'✔' if v else '✘'} {k.replace('_', ' ')}" for k, v in c["resultado"].items())
    return (
        common_header(meta, published, "S3 — Tasa de impago esperada")
        + "## Qué estima\n\nLa tasa acumulada de Over30 a 6 meses de un perfil: `OVER30ACCTS / ENRACCTS` en MOB = 6 "
        "(`OVER30ACCTS` es acumulado). XGBoost con etiqueta fraccional, peso = `ENRACCTS` y restricción monótona decreciente "
        f"en la posición del score (violaciones en prueba: {meta['restriccion_monotona']['violaciones_en_prueba']}). "
        f"Calibración **{cal['metodo']}** ajustada en validación: a = {cal['a']:.4f}, b = {cal['b']:.4f}. {cal['por_que']}\n\n"
        "## Resultados (totales)\n\n" + md_table(totals, fmt) + "\n\n"
        "## Prueba por estrategia\n\n" + md_table(by_s, fmt) + "\n\n"
        "## Prueba por rango de edad\n\n" + md_table(by_a, fmt) + "\n\n"
        "## Niveles de riesgo (prueba)\n\n"
        f"Umbrales: P50 = {u['p50']:.4f}, P80 = {u['p80']:.4f}, P95 = {u['p95']:.4f}. {u['definicion']}\n\n"
        + md_table(levels, lvl_fmt) + "\n\n"
        "## Criterio de publicación\n\n" + c["reglas"] + "\n\n" + checks + f"\n\nResultado: **{'aprobado' if c['aprobado'] else 'no aprobado'}**.\n\n"
        "## Limitaciones\n\n"
        "- DARIA 1.6 solo aparece en validación y prueba: sus métricas miden extrapolación. DARIA 2.0 no es evaluable en riesgo (solo MOB 0 y 1).\n"
        "- La mejora sobre el GLM es pequeña; el poder discriminante (AUC) es moderado en todos los modelos de la base sintética.\n"
        "- Los umbrales de riesgo se calculan aquí para el criterio de publicación; la Fase 3 los usa en las reglas de consistencia.\n"
        "- La herramienta no aprueba ni rechaza créditos: un riesgo Crítico solo genera una alerta de revisión. No afirma causalidad.\n"
    )


# ===== 4. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda el resumen y las fichas."""
    reports = get_path(config, "reports_dir") / "02_modelos_prediccion"
    out = reports / "resumen_modelos.csv"
    results["summary"].to_csv(out, index=False, encoding="utf-8")
    written = [out]
    fichas = PROJECT_ROOT / "docs" / "fichas"
    fichas.mkdir(parents=True, exist_ok=True)
    for key, fn in (("s1", ficha_s1), ("s3", ficha_s3)):
        meta, pub, cmp = results[key]
        if meta is None:
            continue
        path = fichas / f"ficha_{key}.md"
        path.write_text(fn(meta, pub, cmp), encoding="utf-8")
        written.append(path)
    return written


# ===== 5. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 0 si todo salió bien."""
    parse_args()
    config = load_config()
    set_seed(config["seed"])
    results = compute(config)
    written = save_results(results, config)

    s = results["summary"]
    print("Resumen de modelos (totales):")
    for _, row in s.iterrows():
        vals = "  ".join(f"{k} {row[k]:.4g}" for k in s.columns[3:] if pd.notna(row[k]))
        print(f"  {row['salida']} {row['modelo']:31s} {row['conjunto']:11s} {vals}")
    for key in ("s1", "s3"):
        meta, pub, _ = results[key]
        print(f"  {key.upper()}: {'publicado ' + meta['version'] if pub else ('sin versión publicada (última: ' + meta['version'] + ')' if meta else 'sin modelo')}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
