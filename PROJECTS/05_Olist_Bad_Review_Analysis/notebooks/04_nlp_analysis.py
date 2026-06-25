# -*- coding: utf-8 -*-
"""
Project 5: Olist 巴西电商 — Step 4 NLP评论文本分析
核心问题: 差评评论里，消费者到底在骂什么？
"""
import sqlite3
import pandas as pd
import numpy as np
import re
import os, sys
from collections import Counter

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

DB = r"D:\AI_Dateannaly\PROJECTS\project_olist_analysis\data\olist.db"

# 葡萄牙语停用词 + 常用电商词
STOPWORDS = set([
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para',
    'com', 'nao', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais',
    'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'ha', 'foi',
    'meu', 'minha', 'muito', 'so', 'nos', 'ja', 'esta', 'tem',
    'ser', 'fazer', 'quando', 'muito', 'produto', 'produtos',  # produto太泛
    'entregue', 'chegou', 'recebi', 'veio', 'bom', 'bem',       # 太泛
    'pra', 'pro', 'tudo', 'nada', 'ate', 'sem', 'ter', 'sua',
    'pelo', 'pela', 'mim', 'sobre', 'estou', 'entrega',
])

# 关键词映射（葡萄牙语→中文）
KEYWORD_MAP = {
    'entrega': 'delivery',
    'atraso': 'delay', 'atrasou': 'delay', 'atrasado': 'delay',
    'demorou': 'delay', 'demora': 'delay', 'demorado': 'delay',
    'prazo': 'deadline',
    'defeito': 'defect', 'defeituoso': 'defect', 'quebrado': 'broken',
    'estragado': 'damaged', 'danificado': 'damaged',
    'qualidade': 'quality', 'ruim': 'bad', 'pessimo': 'terrible',
    'horrivel': 'terrible',
    'troca': 'exchange', 'devolucao': 'return', 'devolver': 'return',
    'reembolso': 'refund', 'dinheiro': 'money',
    'cor': 'color', 'tamanho': 'size', 'tamanhos': 'size',
    'imagem': 'image', 'foto': 'photo', 'fotos': 'photos',
    'diferente': 'different', 'descrito': 'described', 'descricao': 'description',
    'anuncio': 'advertisement',
    'pagamento': 'payment', 'pago': 'paid', 'preco': 'price',
    'frete': 'shipping_cost', 'transporte': 'transport',
    'atendimento': 'service', 'suporte': 'support',
    'vendedor': 'seller', 'loja': 'store',
    'caixa': 'box', 'embalagem': 'packaging',
    'funciona': 'works', 'funcionando': 'working',
    'comprei': 'bought',
    'entre': 'star_prefix',  # 星级评论的标记
}


def clean(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^a-zà-ú]+', ' ', text)
    return text


def tokenize(text):
    words = text.split()
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def classify_theme(word):
    return KEYWORD_MAP.get(word, None)


def main():
    conn = sqlite3.connect(DB)

    # 读取评论 + 延迟信息
    sql = """
    SELECT
        r.review_score,
        r.review_comment_message,
        r.review_comment_title,
        CAST(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) AS INTEGER) AS delay_days
    FROM reviews r
    JOIN orders o ON r.order_id = o.order_id
    WHERE r.review_comment_message IS NOT NULL
        AND r.review_comment_message != ''
        AND o.order_status = 'delivered'
    """
    df = pd.read_sql(sql, conn)
    conn.close()

    # 分组
    bad = df[df['review_score'] <= 2]
    good = df[df['review_score'] >= 4]
    neutral = df[df['review_score'] == 3]
    delayed_bad = bad[bad['delay_days'] > 0]
    ontime_bad = bad[bad['delay_days'] <= 0]

    print(f"[数据] 有文本评论总数: {len(df):,}")
    print(f"       差评(1-2): {len(bad):,} ({len(bad)/len(df)*100:.1f}%)")
    print(f"       好评(4-5): {len(good):,}")
    print(f"       中评(3): {len(neutral):,}")
    print(f"       差评中延迟: {len(delayed_bad):,} | 差评中非延迟: {len(ontime_bad):,}")

    # ===================================================================
    print("\n" + "=" * 70)
    print("N1: 差评 vs 好评 高频关键词对比")
    print("=" * 70)

    for label, subset in [("差评(1-2星)", bad), ("好评(4-5星)", good)]:
        all_words = []
        for msg in subset['review_comment_message']:
            all_words.extend(tokenize(clean(msg)))
        counter = Counter(all_words)
        top = counter.most_common(20)
        themes = []
        for w, c in top:
            theme = classify_theme(w) or w
            themes.append(f"{w}({theme}) ×{c}")
        print(f"\n  {label}:")
        print(f"   {' | '.join(themes[:15])}")

    # ===================================================================
    print("\n" + "=" * 70)
    print("N2: 差评主题分布 —— 延迟 vs 非延迟差评，骂点有何不同？")
    print("=" * 70)

    theme_groups = {
        'delivery_speed': ['entrega', 'atraso', 'atrasou', 'atrasado', 'demorou', 'demora', 'demorado', 'prazo'],
        'product_quality': ['defeito', 'defeituoso', 'quebrado', 'estragado', 'danificado', 'qualidade', 'ruim', 'pessimo', 'horrivel'],
        'product_match': ['diferente', 'descrito', 'descricao', 'anuncio', 'imagem', 'foto', 'fotos', 'cor', 'tamanho'],
        'service': ['troca', 'devolucao', 'devolver', 'reembolso', 'dinheiro', 'atendimento', 'suporte', 'vendedor', 'loja'],
        'packaging': ['caixa', 'embalagem'],
        'payment': ['pagamento', 'pago', 'preco'],
    }

    for label, subset in [("差评-延迟订单", delayed_bad), ("差评-非延迟订单", ontime_bad)]:
        all_words = []
        for msg in subset['review_comment_message']:
            all_words.extend(tokenize(clean(msg)))
        total = len(all_words)
        print(f"\n  {label} (总词数: {total:,}):")
        for theme_name, keywords in theme_groups.items():
            cnt = sum(all_words.count(k) for k in keywords)
            pct = cnt / total * 100 if total > 0 else 0
            bar = '█' * int(pct * 3)
            print(f"    {theme_name:20s} {cnt:>5} ({pct:4.1f}%) {bar}")

    # ===================================================================
    print("\n" + "=" * 70)
    print("N3: 延迟差评中的具体抱怨样本（随机5条）")
    print("=" * 70)

    sample = delayed_bad[delayed_bad['review_comment_message'].str.len() > 50].sample(
        min(5, len(delayed_bad[delayed_bad['review_comment_message'].str.len() > 50])),
        random_state=42
    )
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        msg = row['review_comment_message'][:200]
        delay = row['delay_days']
        print(f"\n  [{i}] 延迟{days_to_text(delay)}, 评分{row['review_score']}")
        print(f"      {msg}...")

    # ===================================================================
    print("\n" + "=" * 70)
    print("N4: 非延迟差评中的具体抱怨样本（随机5条）—— 不慢为什么也给差评？")
    print("=" * 70)

    sample2 = ontime_bad[ontime_bad['review_comment_message'].str.len() > 50].sample(
        min(5, len(ontime_bad[ontime_bad['review_comment_message'].str.len() > 50])),
        random_state=42
    )
    for i, (_, row) in enumerate(sample2.iterrows(), 1):
        msg = row['review_comment_message'][:200]
        delay = row['delay_days']
        print(f"\n  [{i}] 延迟{days_to_text(delay)}, 评分{row['review_score']}")
        print(f"      {msg}...")

    # ===================================================================
    print("\n" + "=" * 70)
    print("N5: 关键结论 —— 差评原因分层")
    print("=" * 70)

    # 统计包含"延迟类词"的差评占比
    delay_kw = ['atraso', 'atrasou', 'atrasado', 'demorou', 'demora', 'demorado']
    bad_with_delay_words = sum(
        1 for msg in bad['review_comment_message']
        if any(kw in clean(msg).split() for kw in delay_kw)
    )
    bad_from_delay = len(delayed_bad)
    bad_from_other = len(ontime_bad)

    print(f"""
  差评总量:           {len(bad):,}
  其中延迟订单差评:    {bad_from_delay:,} ({bad_from_delay/len(bad)*100:.1f}%)
  其中非延迟订单差评: {bad_from_other:,} ({bad_from_other/len(bad)*100:.1f}%)
  评论中明确提"延迟": {bad_with_delay_words:,} ({bad_with_delay_words/len(bad)*100:.1f}%)
    """)

    print("[完成] NLP分析全部完成")


def days_to_text(d):
    if d <= -10:
        return f"提前{abs(d)}天"
    elif d < 0:
        return f"提前{abs(d)}天"
    elif d == 0:
        return "0天(准时)"
    else:
        return f"+{d}天"


if __name__ == '__main__':
    main()
