from pathlib import Path
import base64
import json
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
UIUX_DATA_DIR = ROOT / "ui-ux-pro-max" / "data"
OUTPUT_DIR = ROOT / "outputs"
VIS_DIR = OUTPUT_DIR / "visualisations"
REPORT_PATH = OUTPUT_DIR / "reports" / "brand_analysis_report.html"


def ensure_nltk_resources():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')


def load_palette():
    default = {
        'primary': '#1E40AF',
        'secondary': '#3B82F6',
        'accent': '#D97706',
        'background': '#F8FAFC',
        'foreground': '#1E3A8A',
        'surface': '#FFFFFF',
        'muted': '#64748B',
        'danger': '#DC2626'
    }
    palette_file = UIUX_DATA_DIR / 'colors.csv'
    if not palette_file.exists():
        return default

    df = pd.read_csv(palette_file)
    selector = df[df['Product Type'].str.contains('Analytics|Dashboard|Social Media|SaaS|Brand', na=False)]
    if selector.shape[0] > 0:
        row = selector.iloc[0]
    else:
        row = df.iloc[0]

    return {
        'primary': row.get('Primary', default['primary']),
        'secondary': row.get('Secondary', default['secondary']),
        'accent': row.get('Accent', default['accent']),
        'background': row.get('Background', default['background']),
        'foreground': row.get('Foreground', default['foreground']),
        'surface': row.get('Card', default['surface']),
        'muted': row.get('Muted Foreground', default['muted']),
        'danger': row.get('Destructive', default['danger'])
    }


def build_wordcloud(text, stop_words, output_path):
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        stopwords=stop_words,
        max_words=120,
        colormap='viridis'
    ).generate(text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, format='png', dpi=150)
    plt.close()


def save_plotly_svg(fig, path):
    try:
        fig.write_image(path, format='svg')
    except Exception:
        fig.write_image(str(path.with_suffix('.png')))


def encode_image(path: Path):
    if not path.exists():
        return ''
    data = base64.b64encode(path.read_bytes()).decode('utf-8')
    mime = 'image/svg+xml' if path.suffix.lower() == '.svg' else 'image/png'
    return f'data:{mime};base64,{data}'


def render_html(summary, palette, trend_html, charts):
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'>
  <title>Brand Monitoring Analysis Report</title>
  <style>
    :root {{
      --bg: {palette['background']};
      --surface: {palette['surface']};
      --primary: {palette['primary']};
      --secondary: {palette['secondary']};
      --accent: {palette['accent']};
      --foreground: {palette['foreground']};
      --muted: {palette['muted']};
      --danger: {palette['danger']};
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--foreground); }}
    .page {{ max-width: 1200px; margin: 0 auto; padding: 32px; }}
    .hero {{ display: grid; gap: 16px; margin-bottom: 32px; }}
    .badge {{ display: inline-flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 999px; background: rgba(59, 130, 246, .15); color: var(--primary); font-weight: 600; }}
    h1 {{ margin: 0; font-size: clamp(2.15rem, 2.5vw, 3.5rem); line-height: 1.05; }}
    h2 {{ margin-top: 48px; margin-bottom: 16px; font-size: 1.75rem; }}
    p {{ line-height: 1.75; margin: 0 0 16px; }}
    .grid-2 {{ display: grid; gap: 24px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
    .card {{ background: var(--surface); border-radius: 24px; box-shadow: 0 18px 48px rgba(15, 23, 42, .08); padding: 24px; }}
    .chart-image {{ width: 100%; border-radius: 18px; }}
    .section-note {{ color: var(--muted); font-size: 0.95rem; }}
    .key-metrics {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 16px; }}
    .metric {{ background: rgba(59, 130, 246, .08); border-radius: 16px; padding: 20px; }}
    .metric strong {{ display: block; font-size: 1.6rem; margin-bottom: 4px; color: var(--foreground); }}
    .metric span {{ color: var(--muted); }}
    .report-footer {{ margin-top: 48px; padding-top: 24px; border-top: 1px solid rgba(15, 23, 42, .08); color: var(--muted); }}
    @media (max-width: 720px) {{ .page {{ padding: 16px; }} }}
  </style>
</head>
<body>
  <div class='page'>
    <div class='hero'>
      <span class='badge'>Brand Monitoring Report</span>
      <div>
        <h1>Brand reputation, emerging issues, and performance insights</h1>
        <p>{summary}</p>
      </div>
    </div>
    <div class='grid-2'>
      <div class='card'>
        <h2>Sentiment snapshot</h2>
        <p class='section-note'>A consolidated view of sentiment across brand mentions and public conversations.</p>
        <img class='chart-image' src='{charts['sentiment_distribution']}' alt='Sentiment distribution'>
      </div>
      <div class='card'>
        <h2>Emerging issues</h2>
        <p class='section-note'>Negative themes and issue drivers identified from customer messages.</p>
        <img class='chart-image' src='{charts['negative_reason']}' alt='Negative issue drivers'>
      </div>
    </div>
    <div class='card' style='margin-top: 24px;'>
      <h2>Sentiment trend</h2>
      <div>{trend_html}</div>
    </div>
    <div class='grid-2' style='margin-top: 24px;'>
      <div class='card'>
        <h2>Topic word cloud</h2>
        <p class='section-note'>Frequently discussed themes in the brand conversation stream.</p>
        <img class='chart-image' src='{charts['topic_wordcloud']}' alt='Topic word cloud'>
      </div>
      <div class='card'>
        <h2>Priority recommendations</h2>
        <ul>
          <li>Monitor negative service incidents with high urgency using sentiment and issue clustering.</li>
          <li>Allocate response resources to customer service failures and flight experience issues first.</li>
          <li>Use sentiment trend breaks to trigger early-warning alerts on real-time dashboards.</li>
          <li>Track campaign impact with daily sentiment ratio and conversation volume.</li>
        </ul>
      </div>
    </div>
    <div class='card' style='margin-top: 24px;'>
      <h2>Appendix</h2>
      <p class='section-note'>This report is generated from the brand monitoring dataset and design palette in the project files.</p>
      <p><strong>Data sources:</strong> <code>data/Tweets.csv</code>. <strong>Design tokens:</strong> <code>ui-ux-pro-max/data/colors.csv</code>.</p>
    </div>
    <div class='report-footer'>
      Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with reusable script and embedded visuals.
    </div>
  </div>
</body>
</html>"""


def main():
    ensure_nltk_resources()
    palette = load_palette()
    tweets_path = DATA_DIR / 'Tweets.csv'
    if not tweets_path.exists():
        raise FileNotFoundError(f'Missing data file: {tweets_path}')

    tweets = pd.read_csv(tweets_path, parse_dates=['tweet_created'])
    tweets['tweet_text'] = tweets['text'].astype(str).fillna('')
    tweets['date'] = pd.to_datetime(tweets['tweet_created'], errors='coerce').dt.date

    sentiment_counts = tweets['airline_sentiment'].value_counts().reindex(['positive', 'neutral', 'negative']).fillna(0)
    sentiment_fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        color=sentiment_counts.index,
        color_discrete_map={
            'positive': palette['secondary'],
            'neutral': palette['primary'],
            'negative': palette['danger']
        },
        labels={'x': 'Sentiment', 'y': 'Mentions'},
        title='Brand Sentiment Distribution'
    )
    sentiment_figure_path = VIS_DIR / 'sentiment_distribution.png'
    sentiment_fig.write_image(str(sentiment_figure_path), width=1000, height=600)

    trend = (
        tweets.dropna(subset=['date'])
        .groupby(['date', 'airline_sentiment'])['tweet_id']
        .count()
        .reset_index(name='count')
    )
    trend_fig = px.area(
        trend,
        x='date',
        y='count',
        color='airline_sentiment',
        color_discrete_map={
            'positive': palette['secondary'],
            'neutral': palette['primary'],
            'negative': palette['danger']
        },
        labels={'count': 'Mentions', 'date': 'Date'},
        title='Daily Sentiment Trend'
    )
    trend_html = pio.to_html(trend_fig, full_html=False, include_plotlyjs='cdn')
    trend_svg_path = VIS_DIR / 'trend_analysis.svg'
    save_plotly_svg(trend_fig, trend_svg_path)

    negative_reasons = (
        tweets.loc[tweets['airline_sentiment'] == 'negative', 'negativereason']
        .fillna('Unknown')
        .value_counts()
        .nlargest(8)
    )
    negative_fig = px.bar(
        x=negative_reasons.values,
        y=negative_reasons.index,
        orientation='h',
        labels={'x': 'Negative Mentions', 'y': 'Issue Type'},
        title='Top Negative Issue Drivers',
        color_discrete_sequence=[palette['danger']] * len(negative_reasons)
    )
    negative_reason_path = VIS_DIR / 'anomalies.png'
    negative_fig.update_layout(margin={'l': 180, 'r': 24, 't': 60, 'b': 24})
    negative_fig.write_image(str(negative_reason_path), width=1000, height=600)

    text_corpus = ' '.join(tweets['tweet_text'].tolist())
    stop_words = set(stopwords.words('english')) | {
        'virginamerica', 'virgin', 'america', 'flight', 'http', 'https', 't.co', 'via', 'rt'
    }
    wordcloud_path = VIS_DIR / 'topic_wordcloud.png'
    build_wordcloud(text_corpus, stop_words, wordcloud_path)

    summary = (
        'This analysis summarizes public sentiment and emerging service issues from social media mentions. '
        'It highlights priority response areas and shows how sentiment evolves over time for the brand.'
    )
    charts = {
        'sentiment_distribution': encode_image(sentiment_figure_path),
        'topic_wordcloud': encode_image(wordcloud_path),
        'negative_reason': encode_image(negative_reason_path)
    }
    html_content = render_html(summary, palette, trend_html, charts)
    REPORT_PATH.write_text(html_content, encoding='utf-8')
    print(f'Generated report: {REPORT_PATH}')
    print(f'Saved visuals to: {VIS_DIR}')


if __name__ == '__main__':
    main()
