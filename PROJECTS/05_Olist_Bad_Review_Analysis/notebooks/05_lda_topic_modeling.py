# -*- coding: utf-8 -*-
"""
Project 5: Olist 巴西电商 — Step 5 LDA主题建模
自动聚类差评评论，发现隐藏模式
"""
import sqlite3
import pandas as pd
import numpy as np
import re
import os, sys

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

DB = r"D:\AI_Dateannaly\PROJECTS\project_olist_analysis\data\olist.db"

# 葡萄牙语停用词
STOPWORDS_PT = [
    'de','a','o','que','e','do','da','em','um','para','com','nao','uma',
    'os','no','se','na','por','mais','as','dos','como','mas','ao','ele',
    'das','ha','foi','meu','minha','muito','so','nos','ja','esta','tem',
    'ser','fazer','quando','produto','produtos','entregue','chegou','recebi',
    'veio','bom','bem','pra','pro','tudo','nada','ate','sem','ter','sua',
    'pelo','pela','mim','sobre','estou','entrega','ainda','apenas','pois',
    'agora','dia','dois','tres','site','comprei','compra','pedido','vou',
    'faz','fiz','fez','vai','era','anos','vez','fiquei','obrigado','obrigada',
    'excelente','otimo','perfeito','recomendo','super','gostei','rapida',
    'dentro','antes','boa','loja','chega','demais','to','ta',
]

def clean_pt(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^a-zà-ú]+', ' ', text)
    words = text.split()
    words = [w for w in words if len(w) > 2 and w not in STOPWORDS_PT]
    return ' '.join(words)


def print_topics(model, feature_names, n_top_words=10):
    topic_map = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        top_weights = [topic[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_map[topic_idx] = top_words
        print(f"\n  Topic #{topic_idx + 1}: {' | '.join(f'{w}({wt:.0f})' for w, wt in zip(top_words, top_weights))}")
    return topic_map


def interpret_topic(words):
    """人工标注主题含义"""
    delivery_kw = {'atraso','atrasou','atrasado','demorou','demora','prazo','correios',
                   'transportadora','entrega','envio','enviado','postagem','chegando'}
    quality_kw = {'defeito','defeituoso','quebrado','estragado','danificado','qualidade',
                  'ruim','pessimo','horrivel','amassados','arranhado','riscado'}
    match_kw   = {'diferente','descrito','descricao','anuncio','imagem','foto','fotos',
                  'cor','tamanho','tamanhos','errado','trocado','modelo'}
    service_kw = {'troca','devolucao','devolver','reembolso','dinheiro','atendimento',
                  'suporte','vendedor','resposta','respondem','resolver','solucao',
                  'contato','email','emails','mensagem'}
    missing_kw = {'faltando','faltou','incompleto','faltaram','metade','apenas','so',
                  'veio','peca','pecas'}
    
    d_score = sum(1 for w in words if w in delivery_kw)
    q_score = sum(1 for w in words if w in quality_kw)
    m_score = sum(1 for w in words if w in match_kw)
    s_score = sum(1 for w in words if w in service_kw)
    mi_score = sum(1 for w in words if w in missing_kw)
    
    scores = {
        '🚚 配送延迟': d_score,
        '🔧 产品质量': q_score,
        '📐 描述不符': m_score,
        '📞 售后无响应': s_score,
        '📦 缺件/漏发': mi_score,
    }
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:2]


def main():
    conn = sqlite3.connect(DB)
    sql = """
    SELECT review_comment_message
    FROM reviews r
    JOIN orders o ON r.order_id = o.order_id
    WHERE r.review_score <= 2
        AND r.review_comment_message IS NOT NULL
        AND r.review_comment_message != ''
        AND o.order_status = 'delivered'
    """
    df = pd.read_sql(sql, conn)
    conn.close()

    print(f"[数据] 差评总数: {len(df):,}")
    
    # 清洗
    df['cleaned'] = df['review_comment_message'].apply(clean_pt)
    df = df[df['cleaned'].str.len() > 10]
    print(f"[数据] 有效评论（清洗后>10字符）: {len(df):,}")

    # 向量化
    print("[处理] 向量化...")
    vec = CountVectorizer(
        max_df=0.5,          # 出现在>50%文档的词忽略
        min_df=10,           # 至少10条评论
        max_features=2000,
        ngram_range=(1, 2),  # 一元+二元词
    )
    dtm = vec.fit_transform(df['cleaned'])
    print(f"[处理] 文档-词矩阵: {dtm.shape[0]:,} docs x {dtm.shape[1]} terms")

    # LDA
    print("[处理] LDA主题建模 (k=6)...")
    lda = LatentDirichletAllocation(
        n_components=6,
        learning_method='online',
        random_state=42,
        max_iter=20,
        batch_size=512,
        n_jobs=-1,
    )
    lda.fit(dtm)

    # 主题关键词
    print("\n" + "=" * 70)
    print("L1: 6个自动发现的差评主题")
    print("=" * 70)
    topic_map = print_topics(lda, vec.get_feature_names_out(), n_top_words=12)

    # 主题占比
    topic_dist = lda.transform(dtm)
    doc_topics = topic_dist.argmax(axis=1)
    topic_counts = pd.Series(doc_topics).value_counts().sort_index()

    print("\n" + "=" * 70)
    print("L2: 各主题占比 & 业务标注")
    print("=" * 70)

    total = len(df)
    for tidx in range(6):
        cnt = topic_counts.get(tidx, 0)
        pct = cnt / total * 100
        bar = '█' * int(pct * 2)
        top_terms = topic_map[tidx][:8]
        labels = interpret_topic(top_terms)
        label_str = ' + '.join([f'{name}({score})' for name, score in labels])
        print(f"  Topic{tidx+1}  {pct:5.1f}% ({cnt:>5,}) {bar}")
        print(f"            关键词: {', '.join(top_terms)}")
        print(f"            标注:   {label_str}")
        print()

    # ===================================================================
    print("=" * 70)
    print("L3: 主题分布 —— 交叉验证延迟/非延迟订单的主题差异")
    print("=" * 70)

    sql2 = """
    SELECT
        r.review_comment_message,
        CAST(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) AS INTEGER) AS delay_days
    FROM reviews r
    JOIN orders o ON r.order_id = o.order_id
    WHERE r.review_score <= 2
        AND r.review_comment_message IS NOT NULL
        AND r.review_comment_message != ''
        AND o.order_status = 'delivered'
    """
    conn2 = sqlite3.connect(DB)
    df2 = pd.read_sql(sql2, conn2)
    conn2.close()
    df2['cleaned'] = df2['review_comment_message'].apply(clean_pt)
    df2 = df2[df2['cleaned'].str.len() > 10]
    df2['is_delayed'] = (df2['delay_days'] > 0).astype(int)

    dtm2 = vec.transform(df2['cleaned'])
    doc_topics2 = lda.transform(dtm2).argmax(axis=1)
    df2['topic'] = doc_topics2

    print(f"\n{'':>30} {'延迟差评':>12} {'非延迟差评':>12} {'集中度':>8}")
    print(f"{'':>30} {'-'*12} {'-'*12} {'-'*8}")
    
    delayed_mask = df2['is_delayed'] == 1
    ontime_mask = df2['is_delayed'] == 0
    total_delayed = delayed_mask.sum()
    total_ontime = ontime_mask.sum()

    for tidx in range(6):
        words = topic_map[tidx][:6]
        label = interpret_topic(words)
        label_str = label[0][0] if label else '???'
        
        d_cnt = ((df2['topic'] == tidx) & delayed_mask).sum()
        o_cnt = ((df2['topic'] == tidx) & ontime_mask).sum()
        d_pct = d_cnt / total_delayed * 100 if total_delayed > 0 else 0
        o_pct = o_cnt / total_ontime * 100 if total_ontime > 0 else 0
        
        # 集中度：延迟占比/非延迟占比
        concentration = d_pct / o_pct if o_pct > 0 else float('inf')
        marker = '← 延迟集中' if concentration > 1.8 else '← 非延迟集中' if concentration < 0.5 else ''
        print(f"  Topic{tidx+1} {label_str:20s} {d_pct:5.1f}% ({d_cnt:>4,}) {o_pct:5.1f}% ({o_cnt:>4,})  {concentration:3.1f}x {marker}")

    print("\n[完成] LDA主题建模完成")


if __name__ == '__main__':
    main()
