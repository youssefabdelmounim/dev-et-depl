"""Streamlit interface for EduAccess.

Run (after the API is up)::

    streamlit run app/streamlit_app.py

The UI only talks to the FastAPI service (``EDUACCESS_API_URL``, default
``http://127.0.0.1:8000``), so it can be deployed on a different host.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

API_URL = os.environ.get("EDUACCESS_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="EduAccess - Détection précoce des difficultés",
    page_icon="🎓",
    layout="wide",
)

RISK_COLORS = {"high": "#d9534f", "medium": "#f0ad4e", "low": "#5cb85c"}
RISK_LABELS_FR = {"high": "Risque élevé", "medium": "Risque intermédiaire",
                  "low": "Risque faible"}


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    try:
        response = httpx.get(f"{API_URL}{path}", params=params, timeout=60.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API {path} → HTTP {exc.response.status_code} : "
                 f"{exc.response.json().get('detail', exc.response.text)}")
    except httpx.HTTPError:
        st.error(
            "Impossible de joindre l'API. Lancez-la d'abord :\n\n"
            f"`uvicorn api.main:app --reload` (URL attendue : {API_URL})"
        )
    return None


def api_post(path: str, json_body: Optional[Dict] = None,
             params: Optional[Dict[str, Any]] = None) -> Any:
    try:
        response = httpx.post(f"{API_URL}{path}", json=json_body, params=params,
                              timeout=60.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API {path} → HTTP {exc.response.status_code} : "
                 f"{exc.response.json().get('detail', exc.response.text)}")
    except httpx.HTTPError:
        st.error(
            "Impossible de joindre l'API. Lancez-la d'abord :\n\n"
            f"`uvicorn api.main:app --reload` (URL attendue : {API_URL})"
        )
    return None


def show_figure(name: str, caption: str) -> None:
    try:
        response = httpx.get(f"{API_URL}/figures/{name}", timeout=60.0)
        if response.status_code == 200:
            st.image(response.content, caption=caption, use_container_width=True)
    except httpx.HTTPError:
        pass


# --------------------------------------------------------------------------- #
# Prediction rendering
# --------------------------------------------------------------------------- #
def render_prediction(result: Dict[str, Any]) -> None:
    probability = result["probability_at_risk"]
    level = result["risk_level"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Classe prédite",
                "À risque" if result["prediction"] == 1 else "Succès")
    col2.metric("P(at-risque)", f"{probability:.1%}")
    col3.metric("Niveau", RISK_LABELS_FR.get(level, level))
    st.progress(min(max(probability, 0.0), 1.0),
                text=f"Probabilité d'être en difficulté : {probability:.1%}")
    color = RISK_COLORS.get(level, "#337ab7")
    st.markdown(
        f"<div style='padding:0.8rem;border-left:6px solid {color};"
        f"background:{color}15;margin-top:0.5rem'>"
        f"<b>{RISK_LABELS_FR.get(level, level)}</b> — les valeurs manquantes "
        f"sont imputées comme à l'entraînement "
        f"({result['provided_features']}/{result['total_features']} fournies, "
        f"{len(result.get('imputed_features', []))} imputées). "
        f"Modèle : <code>{result['model_name']}</code>.</div>",
        unsafe_allow_html=True,
    )
    if result.get("imputed_features"):
        with st.expander("Colonnes imputées (non fournies)"):
            st.write(", ".join(result["imputed_features"]))


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_indicators() -> None:
    st.header("Indicateurs du jeu de données (OULAD)")
    st.caption("32 593 inscriptions de l'Open University sur 22 présentations "
               "de 7 modules — la cible binaire « à risque » regroupe les "
               "résultats Fail et Withdrawn.")
    data = api_get("/indicators")
    if not data:
        return

    overview = data["overview"]
    kpi = st.columns(4)
    kpi[0].metric("Inscriptions", f"{overview['n_students']:,}".replace(",", " "))
    kpi[1].metric("Modules / présentations",
                  f"{overview['n_modules']} / {overview['n_presentations']}")
    kpi[2].metric("Taux à risque", f"{overview['at_risk_rate']:.1%}")
    kpi[3].metric("Modèle en production", data.get("model", {}).get("best_model", "—"))

    left, right = st.columns(2)
    with left:
        st.subheader("Répartition des résultats finaux")
        st.bar_chart({row["final_result"]: row["n"]
                      for row in data["final_result_distribution"]})
    with right:
        st.subheader("Engagement précoce : à risque vs succès")
        compare = data["group_comparison"]
        rows = []
        for feature, values in compare.items():
            rows.append({
                "Indicateur": feature.replace("vle_", "VLE ").replace("assess_", "Éval. "),
                "Succès": values["success"],
                "À risque": values["at_risk"],
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Taux de risque selon le profil")
    tabs = st.tabs(["Formation", "IMD (quartier)", "Région", "Module",
                    "Genre", "Handicap", "Âge"])
    panels = [
        ("at_risk_by_education", "highest_education"),
        ("at_risk_by_imd_band", "imd_band"),
        ("at_risk_by_region", "region"),
        ("at_risk_by_module", "code_module"),
        ("at_risk_by_gender", "gender"),
        ("at_risk_by_disability", "disability"),
        ("at_risk_by_age_band", "age_band"),
    ]
    for tab, (key, label) in zip(tabs, panels):
        with tab:
            frame_rows = data[key]
            st.bar_chart({r.get(label, "Unknown"): r["at_risk_rate"]
                          for r in reversed(frame_rows)})
            st.dataframe(frame_rows, use_container_width=True, hide_index=True)

    st.subheader("Distribution de l'engagement")
    engagement = data["engagement_distribution"]
    cols = st.columns(len(engagement))
    labels = {
        "vle_clicks_30d_median": "Clics VLE 30j (médiane)",
        "vle_clicks_30d_mean": "Clics VLE 30j (moyenne)",
        "vle_active_days_30d_mean": "Jours actifs 30j (moyenne)",
        "no_vle_activity_30d_share": "Part sans activité VLE 30j",
        "submit_rate_60d_mean": "Taux de remise 60j (moyenne)",
    }
    for col, (key, value) in zip(cols, engagement.items()):
        shown = f"{value:.1%}" if "share" in key or "rate" in key else value
        col.metric(labels.get(key, key), shown)


def page_prediction() -> None:
    st.header("Tester une prédiction")
    st.caption("Outil d'aide à l'analyse : une prédiction n'est jamais une "
               "décision automatique concernant l'étudiant.")

    tab_real, tab_manual = st.tabs(["Étudiant réel du jeu de données",
                                    "Saisie manuelle"])

    with tab_real:
        st.write("Sélectionnez une inscription réelle ; le modèle n'utilise "
                 "que les informations disponibles dans les premières semaines.")
        sample = api_get("/students/sample", params={"n": 200})
        if sample:
            options = {
                f"{s['code_module']} {s['code_presentation']} — étudiant "
                f"{s['id_student']} ({s['region']}, {s['highest_education']})": s
                for s in sample["students"]
            }
            choice = st.selectbox("Inscription", list(options.keys()))
            student = options[choice]
            if st.button("Prédire", type="primary", key="predict_real"):
                result = api_post(
                    f"/predict/student/{student['id_student']}",
                    params={
                        "code_module": student["code_module"],
                        "code_presentation": student["code_presentation"],
                    },
                )
                if result:
                    render_prediction(result)
                    st.info(f"Résultat réel observé : **{student['final_result']}** "
                            f"(à risque = {bool(student['at_risk'])}) — utile pour "
                            "vérifier la cohérence de la prédiction.")

    with tab_manual:
        st.write("Renseignez les informations connues ; tout champ non renseigné "
                 "est imputé (médiane ou catégorie « Missing ») comme pendant "
                 "l'entraînement.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            module = st.selectbox("Module", ["AAA", "BBB", "CCC", "DDD", "EEE",
                                             "FFF", "GGG"])
            presentation = st.selectbox("Présentation", [
                "2013B", "2013J", "2014B", "2014J"])
            gender = st.selectbox("Genre", ["M", "F"])
        with c2:
            region = st.selectbox("Région", [
                "East Anglian Region", "Scotland", "North Western Region",
                "South East Region", "West Midlands Region", "Wales",
                "North Region", "Ireland", "South Region", "Yorkshire Region",
                "East Midlands Region", "London Region", "South West Region"])
            education = st.selectbox("Plus haut diplôme", [
                "A Level or Equivalent", "Lower Than A Level",
                "HE Qualification", "Post Graduate Qualification",
                "No Formal quals"])
        with c3:
            imd = st.selectbox("Bande IMD", [
                "", "0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%",
                "60-70%", "70-80%", "80-90%", "90-100%"])
            age = st.selectbox("Tranche d'âge", ["0-35", "35-55", "55<="])
            disability = st.selectbox("Handicap", ["N", "Y"])
        with c4:
            date_reg = st.number_input("Inscription (jours avant le début, négatif)",
                                       value=-60, step=1)
            prev = st.number_input("Tentatives précédentes", value=0, min_value=0,
                                   step=1)
            credits = st.number_input("Crédits étudiés", value=60, min_value=0,
                                      step=10)

        e1, e2, e3 = st.columns(3)
        with e1:
            clicks30 = st.number_input("Clics VLE (30 premiers jours)", value=40,
                                       min_value=0, step=1)
        with e2:
            days30 = st.number_input("Jours d'activité VLE (30j)", value=10,
                                     min_value=0, step=1)
        with e3:
            rate60 = st.slider("Taux de remise des évaluations (60j)", 0.0, 1.0, 1.0)

        a1, a2 = st.columns(2)
        with a1:
            submitted60 = st.number_input("Évaluations remises (60j)", value=3,
                                          min_value=0, step=1)
        with a2:
            mean_score = st.number_input("Note moyenne des évaluations remises",
                                         value=70, min_value=0, max_value=100,
                                         step=1)

        if st.button("Prédire", type="primary", key="predict_manual"):
            features = {
                "code_module": module,
                "code_presentation": presentation,
                "gender": gender,
                "region": region,
                "highest_education": education,
                "imd_band": imd if imd else None,
                "age_band": age,
                "disability": disability,
                "date_registration": date_reg,
                "num_of_prev_attempts": prev,
                "studied_credits": credits,
                "vle_clicks_30d": clicks30,
                "vle_active_days_30d": days30,
                "assess_submit_rate_60d": rate60,
                "assess_submitted_60d": submitted60,
                "assess_mean_score_60d": mean_score,
            }
            result = api_post("/predict", json_body={"features": features})
            if result:
                render_prediction(result)


def page_model() -> None:
    st.header("Modèle : comparaison et évaluation")
    info = api_get("/model/info")
    if info:
        st.subheader("Modèle déployé")
        cols = st.columns(3)
        cols[0].metric("Algorithme", info["model_name"].replace("_", " ").title())
        cols[1].metric("Features", info["n_features"])
        cols[2].metric("Entraîné le", info["trained_at"][:10])

        st.subheader("Comparaison des trois modèles (jeu de test, 20 %)")
        rows = []
        names = {
            "logistic_regression": "Régression logistique",
            "random_forest": "Forêt aléatoire",
            "hist_gradient_boosting": "Gradient Boosting",
        }
        for name, m in info["all_metrics"].items():
            rows.append({
                "Modèle": names.get(name, name),
                "Accuracy": m["accuracy"],
                "Précision": m["precision"],
                "Rappel": m["recall"],
                "F1": m["f1"],
                "ROC-AUC": m["roc_auc"],
                "F1 CV (moy ± éct)": f"{m['cv_f1_mean']:.3f} ± {m['cv_f1_std']:.3f}",
                "Temps (s)": m["train_time_s"],
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        best = info["all_metrics"][info["model_name"]]
        st.caption(
            f"Modèle retenu : **{names.get(info['model_name'])}** — "
            f"F1 = {best['f1']:.3f}, ROC-AUC = {best['roc_auc']:.3f}. "
            "Le rappel (détection des étudiants à risque) est la métrique "
            "prioritaire pour une détection précoce."
        )

    st.subheader("Courbes et matrices")
    c1, c2 = st.columns(2)
    with c1:
        show_figure("roc_comparison.png", "Courbes ROC des trois modèles")
    with c2:
        show_figure("confusion_hist_gradient_boosting.png",
                    "Matrice de confusion du modèle retenu")
    c3, c4 = st.columns(2)
    with c3:
        show_figure("precision_recall.png", "Précision / Rappel (classe à risque)")
    with c4:
        show_figure("feature_importance.png",
                    "Importance par permutation des features")


def page_about() -> None:
    st.header("À propos d'EduAccess")
    st.markdown(
        """
**EduAccess — détection précoce des difficultés d'apprentissage**

L'application estime, à partir du profil d'inscription et de l'engagement des
**premières semaines** (VLE et évaluations remises jusqu'au jour 60), la
probabilité qu'un étudiant présente des signes de difficulté académique
(*Fail* ou *Withdrawn* au final).

### Architecture
```
dataset/OULAD (CSV) ──► src/ (features) ──► artifacts/ (modèle + métriques)
                                               │
                     UI Streamlit ◄── HTTP ──► API FastAPI
```

### Garde-fous anti-leakage
* aucune donnée postérieure au jour 60 (fenêtre d'observation) ;
* `date_unregistration` exclue (elle révèle le retrait = cible) ;
* notes finales et `final_result` jamais utilisées comme features ;
* imputation et encodage appris **uniquement** sur le train.

### Limites
* outil d'aide à l'analyse, pas une décision automatique ;
* biais possibles liés aux données (IMD manquant pour ~3 % des inscriptions) ;
* les fenêtres 14/30/60 jours sont des choix pédagogiques, pas optimisés.

### Technologies
Python, pandas, scikit-learn, FastAPI, Streamlit, pytest.
"""
    )
    health = api_get("/health")
    if health:
        st.json(health)


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #
st.sidebar.title("🎓 EduAccess")
st.sidebar.caption("Détection précoce des difficultés d'apprentissage — "
                   "Open University Learning Analytics Dataset")
page = st.sidebar.radio(
    "Navigation",
    ["Indicateurs", "Tester une prédiction", "Modèle & métriques", "À propos"],
)
st.sidebar.divider()
st.sidebar.markdown(f"API : `{API_URL}`")
health = api_get("/health")
if health:
    ok = health["model_loaded"] and health["features_loaded"]
    if ok:
        st.sidebar.success("API connectée · modèle chargé")
    else:
        st.sidebar.error("API connectée mais artefacts manquants — "
                         "exécutez `python -m src.train`")

PAGES = {
    "Indicateurs": page_indicators,
    "Tester une prédiction": page_prediction,
    "Modèle & métriques": page_model,
    "À propos": page_about,
}
PAGES[page]()
