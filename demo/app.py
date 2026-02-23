"""Streamlit demo for IslamicNER API."""

from __future__ import annotations

import html
import os
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
GITHUB_REPO_URL = os.getenv("GITHUB_REPO_URL", "https://github.com/yourusername/islamic-ner")

ENTITY_COLORS = {
    "SCHOLAR": "#1d4ed8",  # blue
    "BOOK": "#15803d",  # green
    "CONCEPT": "#c2410c",  # orange
    "PLACE": "#b91c1c",  # red
    "HADITH_REF": "#7e22ce",  # purple
}

SAMPLE_TEXTS = {
    "Bukhari #1 (Intentions)": (
        "حدثنا الحميدي عبد الله بن الزبير قال حدثنا سفيان قال حدثنا يحيى بن سعيد الأنصاري "
        "قال أخبرني محمد بن إبراهيم التيمي أنه سمع علقمة بن وقاص الليثي يقول "
        "سمعت عمر بن الخطاب رضي الله عنه على المنبر يقول سمعت رسول الله صلى الله عليه وسلم يقول "
        "إنما الأعمال بالنيات."
    ),
    "Isnad Chain Example": "حدثنا عبد الله عن مالك عن نافع عن ابن عمر قال قال رسول الله صلى الله عليه وسلم.",
    "Concepts (Riba and Salah)": "نهى النبي صلى الله عليه وسلم عن الربا وأمر بإقامة الصلاة وإيتاء الزكاة.",
    "Book Reference Example": "روى البخاري في صحيح البخاري وروى مسلم في صحيحه هذا الحديث.",
}


def call_api(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        f"{API_URL}{path}",
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def highlight_entities(text: str, entities: List[Dict[str, Any]]) -> str:
    if not text:
        return '<div class="arabic" dir="rtl"></div>'

    safe_entities = []
    for entity in entities:
        try:
            start = int(entity.get("start", -1))
            end = int(entity.get("end", -1))
            if start < 0 or end <= start:
                continue
            safe_entities.append(
                {
                    "start": start,
                    "end": end,
                    "type": str(entity.get("type", "")),
                }
            )
        except (TypeError, ValueError):
            continue

    safe_entities.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))

    chunks: List[str] = []
    cursor = 0
    for entity in safe_entities:
        start = entity["start"]
        end = entity["end"]
        if start < cursor:
            continue

        chunks.append(html.escape(text[cursor:start]))
        color = ENTITY_COLORS.get(entity["type"], "#4b5563")
        entity_text = html.escape(text[start:end])
        chunks.append(
            (
                f'<span style="background:{color};color:white;padding:3px 7px;'
                f'border-radius:8px;margin:0 2px;">{entity_text}</span>'
            )
        )
        cursor = end

    chunks.append(html.escape(text[cursor:]))
    return f'<div class="arabic" dir="rtl">{"".join(chunks)}</div>'


def relation_rows(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for relation in relations:
        source = relation.get("source", {})
        target = relation.get("target", {})
        rows.append(
            {
                "Source": source.get("text", ""),
                "Relation": relation.get("type", ""),
                "Target": target.get("text", ""),
                "Confidence": round(float(relation.get("confidence", 0.0)), 3),
            }
        )
    return rows


def render_graph(entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> None:
    graph = nx.DiGraph()

    def node_id(entity: Dict[str, Any]) -> str:
        return f"{entity.get('type', 'UNKNOWN')}::{entity.get('text', '')}"

    for entity in entities:
        nid = node_id(entity)
        graph.add_node(
            nid,
            label=str(entity.get("text", "")),
            entity_type=str(entity.get("type", "UNKNOWN")),
        )

    for relation in relations:
        source = relation.get("source", {})
        target = relation.get("target", {})
        source_id = node_id(source)
        target_id = node_id(target)

        if source_id not in graph:
            graph.add_node(
                source_id,
                label=str(source.get("text", "")),
                entity_type=str(source.get("type", "UNKNOWN")),
            )
        if target_id not in graph:
            graph.add_node(
                target_id,
                label=str(target.get("text", "")),
                entity_type=str(target.get("type", "UNKNOWN")),
            )

        graph.add_edge(source_id, target_id, relation_type=str(relation.get("type", "")))

    if graph.number_of_nodes() == 0:
        st.info("No graphable entities found in this text.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    pos = nx.spring_layout(graph, seed=42, k=1.2)

    node_colors = [
        ENTITY_COLORS.get(graph.nodes[node].get("entity_type", "UNKNOWN"), "#4b5563")
        for node in graph.nodes
    ]
    node_labels = {node: graph.nodes[node].get("label", node) for node in graph.nodes}
    edge_labels = {(u, v): attrs.get("relation_type", "") for u, v, attrs in graph.edges(data=True)}

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1700, alpha=0.92, ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="#334155",
        width=1.8,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=9, font_color="#0f172a", ax=ax)
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    ax.set_axis_off()
    st.pyplot(fig, use_container_width=True)


st.set_page_config(
    page_title="IslamicNER — Arabic Islamic Text Analysis",
    layout="wide",
    page_icon="🕌",
)

st.markdown(
    """
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Amiri&display=swap");
    .arabic {
        font-family: "Amiri", serif;
        direction: rtl;
        text-align: right;
        font-size: 1.3em;
        line-height: 2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("IslamicNER Demo")
st.sidebar.write(
    "This demo analyzes Arabic Islamic text with an AraBERT-based NER model, "
    "extracts key entities, and optionally builds relations for a knowledge graph."
)
st.sidebar.markdown(f"[GitHub Repository]({GITHUB_REPO_URL})")
st.sidebar.markdown("**Model:** AraBERT v02, fine-tuned on Islamic NER")
st.sidebar.markdown("**Gold F1:** 91.32%")
st.sidebar.markdown("**Entity types:** SCHOLAR, BOOK, CONCEPT, PLACE, HADITH_REF")
st.sidebar.warning("Disclaimer: This is an NLP research tool, not a religious authority.")

st.title("IslamicNER — Arabic Islamic Text Analysis")
st.caption("Paste a hadith or classical Arabic text, then inspect entities, relations, and graph structure.")

if "input_text" not in st.session_state:
    st.session_state.input_text = SAMPLE_TEXTS["Bukhari #1 (Intentions)"]
if "ner_result" not in st.session_state:
    st.session_state.ner_result = None
if "graph_result" not in st.session_state:
    st.session_state.graph_result = None
if "graph_error" not in st.session_state:
    st.session_state.graph_error = ""

selected_sample = st.selectbox("Sample Texts", list(SAMPLE_TEXTS.keys()))
if st.button("Load Sample"):
    st.session_state.input_text = SAMPLE_TEXTS[selected_sample]
    st.rerun()

st.text_area("Arabic Text Input", key="input_text", height=230)

if st.button("Analyze", type="primary"):
    text = st.session_state.input_text.strip()
    if not text:
        st.warning("Please provide Arabic text before analysis.")
    else:
        st.session_state.graph_error = ""
        try:
            st.session_state.ner_result = call_api("/ner", {"text": text, "return_tokens": False})
        except requests.RequestException as exc:
            st.session_state.ner_result = None
            st.session_state.graph_result = None
            st.error(f"Could not reach /ner endpoint at {API_URL}. Details: {exc}")
        else:
            try:
                st.session_state.graph_result = call_api("/graph/build", {"text": text})
            except requests.RequestException as exc:
                st.session_state.graph_result = None
                st.session_state.graph_error = str(exc)

ner_result = st.session_state.ner_result
graph_result = st.session_state.graph_result
graph_error = st.session_state.graph_error

if ner_result:
    entities = ner_result.get("entities", [])
    normalized_text = ner_result.get("normalized_text", "")

    tab_entities, tab_relations, tab_graph = st.tabs(["Entities", "Relations", "Graph"])

    with tab_entities:
        st.subheader("Highlighted Text")
        st.markdown(highlight_entities(normalized_text, entities), unsafe_allow_html=True)

        entity_rows = [
            {
                "Text": entity.get("text", ""),
                "Type": entity.get("type", ""),
                "Confidence": round(float(entity.get("confidence", 0.0)), 3),
            }
            for entity in entities
        ]
        if entity_rows:
            st.dataframe(pd.DataFrame(entity_rows), use_container_width=True)
        else:
            st.info("No entities detected for this text.")

    with tab_relations:
        if graph_result:
            relations = graph_result.get("relations", [])
            st.metric("Nodes Inserted", int(graph_result.get("nodes_inserted", 0)))
            st.metric("Relations Inserted", int(graph_result.get("relations_inserted", 0)))

            if relations:
                grouped = defaultdict(list)
                for row in relation_rows(relations):
                    grouped[row["Relation"]].append(row)

                for relation_type in sorted(grouped):
                    st.markdown(f"**{relation_type}**")
                    st.dataframe(pd.DataFrame(grouped[relation_type]), use_container_width=True)
            else:
                st.info("No relations extracted from this text.")
        else:
            st.info("Relation extraction/graph insertion is unavailable right now. Showing entities only.")
            if graph_error:
                st.caption(f"Graph endpoint error: {graph_error}")

    with tab_graph:
        if graph_result:
            render_graph(
                entities=graph_result.get("entities", entities),
                relations=graph_result.get("relations", []),
            )
        else:
            render_graph(entities=entities, relations=[])
            if graph_error:
                st.caption(f"Graph endpoint error: {graph_error}")
