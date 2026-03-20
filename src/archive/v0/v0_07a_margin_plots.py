import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# Carica il dataset con i margini (già prodotto dallo script 07 v3)
merged = pd.read_csv("data/dataset_merged_with_futures.csv", index_col=0, parse_dates=True)

WAR_EVENTS = {
    "Ucraina": ("2022-02-24", "#e74c3c"),
    "Iran-Israele": ("2025-06-13", "#e67e22"),
    "Hormuz": ("2026-02-28", "#8e44ad"),
}

DPI = 180

def plot_margin(series_name, col, color, title):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(merged.index, merged[col], color=color, lw=2.2, label=series_name)
    
    # Eventi di guerra
    for label, (date, c) in WAR_EVENTS.items():
        ts = pd.Timestamp(date)
        if merged.index[0] <= ts <= merged.index[-1]:
            ax.axvline(ts, color=c, lw=1.8, linestyle="--", alpha=0.9)
            ax.text(ts + pd.Timedelta(days=5), merged[col].max() * 0.97,
                    label, rotation=90, fontsize=9, color=c, va="top")
    
    ax.set_ylabel("Margine lordo (EUR/litro)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = f"plots/01d_margine_{series_name.lower()}.png"
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ Salvato: {fname}")

# Genera i due grafici
plot_margin("Benzina", "margine_benzina", "#d6604d",
            "Margine lordo Benzina Italia — 2021-2026\n(prezzi senza tasse)")
plot_margin("Diesel",  "margine_diesel",  "#31a354",
            "Margine lordo Diesel Italia — 2021-2026\n(prezzi senza tasse)")

print("Fatto! Ora hai i grafici 01d e 01e per i margini (stesso stile di 01a-01b-01c).")