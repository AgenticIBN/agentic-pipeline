import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Güncellenmiş Final Veriler
data = {
    "Model": ["Qwen 0.5B", "Qwen 1.5B", "Gemma 2B", "Llama 3B", "Qwen 7B"],
    "Accuracy": [28.9, 34.5, 36.9, 46.1, 44.6],      # Yüzde
    "Latency": [22.26, 19.09, 31.15, 23.39, 32.34],  # Saniye
    "Energy": [490, 420, 685, 515, 711]              # Joule
}

df = pd.DataFrame(data)

# Grafik Ayarları
fig, ax1 = plt.subplots(figsize=(12, 7))

# Renkler
color_acc = '#2ca02c' # Yeşil (Accuracy)
color_lat = '#1f77b4' # Mavi (Latency)
# Şampiyonu (Llama 3B) vurgulamak için özel renk
highlight_color = '#ff7f0e' 

# Bar Genişliği
bar_width = 0.35
index = np.arange(len(df["Model"]))

# 1. Bar: Accuracy (Sol Eksen)
# Llama 3B'yi turuncu yapalım
acc_colors = [color_acc if m != "Llama 3B" else highlight_color for m in df["Model"]]
bars1 = ax1.bar(index, df["Accuracy"], bar_width, label='Accuracy (%)', color=acc_colors, alpha=0.9)

ax1.set_xlabel('Edge LLM Models', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', color=color_acc, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_acc)
ax1.set_ylim(0, 60) 

# İkinci Eksen (Sağ Taraf - Latency)
ax2 = ax1.twinx() 
# Latency barları şeffaf olsun ki arkası görünsün
bars2 = ax2.bar(index + bar_width, df["Latency"], bar_width, label='Latency (s)', color=color_lat, alpha=0.5)

ax2.set_ylabel('Latency (s)', color=color_lat, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_lat)
ax2.set_ylim(0, 40)

# X ekseni
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(df["Model"])

# Başlık
plt.title('Performance Analysis: Finding the Optimal Edge Model', fontsize=14)

# Barların üzerine değerleri yazalım
def add_labels(bars, ax, format_str='{:.1f}'):
    for rect in bars:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.5,
                format_str.format(height),
                ha='center', va='bottom', fontsize=9, fontweight='bold')

add_labels(bars1, ax1, '{:.1f}%')

# Enerji Değerlerini Latency barlarının içine yazalım
for i, rect in enumerate(bars2):
    height = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., height/2, 
             f'{int(df["Energy"][i])}J',
             ha='center', va='center', color='white', fontweight='bold', fontsize=10, rotation=90)

# Legend (Açıklama)
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=color_acc, lw=4),
                Line2D([0], [0], color=color_lat, alpha=0.5, lw=4),
                Line2D([0], [0], color=highlight_color, lw=4)]
ax1.legend(custom_lines, ['Accuracy', 'Latency', 'Sweet Spot (Llama 3B)'], loc='upper left')

plt.tight_layout()
plt.savefig("final_paper_graph.png", dpi=300)
plt.show()

print("Grafik 'final_paper_graph.png' olarak kaydedildi.")
