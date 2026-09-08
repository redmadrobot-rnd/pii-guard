"""Streamlit-демо: обезличивание и обратное восстановление.

    pip install streamlit
    streamlit run examples/demo_app.py

Работает с библиотекой напрямую, без HTTP-сервиса.
"""

from __future__ import annotations

import streamlit as st

from pii_guard import Anonymizer

EXAMPLE = (
    "Меня зовут Иван Петров, ИНН 7707083893, СНИЛС 112-233-445 95.\n"
    "Телефон +7 916 123-45-67, почта ivan.petrov@example.com.\n"
    "Живу в Москве, ул. Садовая, д. 12, индекс 101000."
)


@st.cache_resource(show_spinner="Загружаю модели…")
def get_anonymizer() -> Anonymizer:
    return Anonymizer()


st.set_page_config(page_title="pii-guard", page_icon="🛡", layout="wide")
st.title("pii-guard")
st.caption("Обезличивание ПДн в русском тексте с обратимостью")

anonymizer = get_anonymizer()
if not anonymizer.ner_enabled:
    st.warning(
        "NER отключён — работает только ветка правил. "
        "PERSON, LOCATION, PHONE_NUMBER и URL находиться не будут."
    )

mode = st.radio(
    "Режим",
    ["pseudonymize", "mask", "tag"],
    horizontal=True,
    help="pseudonymize — единственный обратимый режим",
)
text = st.text_area("Исходный текст", EXAMPLE, height=160)

if st.button("Обезличить", type="primary"):
    result = anonymizer.anonymize(text, mode=mode)
    st.session_state["result"] = result

result = st.session_state.get("result")
if result:
    left, right = st.columns(2)
    with left:
        st.subheader("Обезличено")
        st.code(result.text, language=None, wrap_lines=True)
    with right:
        st.subheader(f"Найдено сущностей: {len(result.entities[0])}")
        st.dataframe(
            [
                {
                    "тип": e["entity_type"],
                    "значение": result.normalized_texts[0][e["start"]: e["end"]],
                    "score": round(e["score"], 3) if e.get("score") else None,
                }
                for e in sorted(result.entities[0], key=lambda e: e["start"])
            ],
            use_container_width=True,
            hide_index=True,
        )

    if result.mapping:
        st.divider()
        st.subheader("Восстановление")
        st.caption(
            "Ответ модели с плейсхолдерами. Имена и адреса встанут в падеж, "
            "который требует контекст."
        )
        # Именно PERSON: падеж виден только на нём, а ключи таблицы идут в
        # обратном порядке относительно текста, поэтому `next(iter(...))` дал бы
        # последнюю сущность — как правило LOCATION.
        person_tag = next(
            (tag for tag in result.mapping if 'type="PERSON"' in tag),
            next(iter(result.mapping)),
        )
        answer = st.text_area(
            "Ответ модели",
            f"Свяжитесь с {person_tag}, я уже написал ему.",
            height=90,
            key="llm_answer",
        )
        if st.button("Восстановить"):
            st.code(anonymizer.deanonymize(answer, result.mapping)[0], language=None)

        with st.expander("Таблица псевдонимов (нигде не хранится)"):
            st.json(result.mapping)
