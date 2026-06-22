import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Verileri elle oluşturuyoruz (Hesapladığımız enerji değerleriyle)
data = {
    "Model": ["Qwen 0.5B", "Qwen 1.5B", "Llama 1B", "Gemma 2B"],
    "Accuracy": [28.9, 34.5, 34.1, 36.9],      # Yüzde cinsinden
    "Latency": [22.26, 19.09, 27.33, 31.15],   # Saniye
    "Energy": [489.7, 419.9, 601.2, 685.3]     # Joule
}

df = pd.DataFrame(data)

# Grafik Ayarları
fig, ax1 = plt.subplots(figsize=(10, 6))

# Renkler
color_acc = '#2ca02c' # Yeşil
color_lat = '#1f77b4' # Mavi
color_enr = '#d62728' # Kırmızı

# Bar Genişliği
bar_width = 0.25
index = np.arange(len(df["Model"]))

# 1. Bar: Accuracy (Sol Eksen)
bars1 = ax1.bar(index, df["Accuracy"], bar_width, label='Accuracy (%)', color=color_acc, alpha=0.7)
ax1.set_xlabel('Mini-LLM Models', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', color=color_acc, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_acc)
ax1.set_ylim(0, 50) # Accuracy skalası

# İkinci Eksen (Sağ Taraf - Latency ve Energy için)
ax2 = ax1.twinx() 

# 2. Bar: Latency
bars2 = ax2.bar(index + bar_width, df["Latency"], bar_width, label='Latency (s)', color=color_lat, alpha=0.7)

# 3. Bar: Energy (Enerjiyi ölçekleyerek aynı eksene sığdıralım veya sadece Latency gösterelim)
# Karmaşıklığı önlemek için Latency ve Energy korele olduğu için Latency'yi gösterip Energy'yi text olarak yazabiliriz.
ax2.set_ylabel('Latency (s)', color=color_lat, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_lat)
ax2.set_ylim(0, 40)

# X ekseni etiketleri
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(df["Model"])

# Başlık
plt.title('Trade-off Analysis: Accuracy vs Latency on Edge Device (M3 Pro)', fontsize=14)

# Enerji değerlerini barların üzerine yazalım
for i, rect in enumerate(bars2):
    height = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., height + 0.5,
             f'{int(df["Energy"][i])}J',
             ha='center', va='bottom', color=color_enr, fontweight='bold')

# Legend (Açıklama)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
plt.savefig("benchmark_graph.png", dpi=300)
plt.show()

print("Grafik 'benchmark_graph.png' olarak kaydedildi.")