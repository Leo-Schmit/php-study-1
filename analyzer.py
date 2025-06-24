import json
from collections import Counter
import matplotlib.pyplot as plt

with open('combined_repo_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

all_exists = []
for repo in data.values():
    all_exists.extend(repo.get('exists_analyzer', []))

counter = Counter(all_exists)

labels = list(counter.keys())
sizes = list(counter.values())

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 14}
)

for text in texts:
    text.set_fontsize(14)
for autotext in autotexts:
    autotext.set_fontsize(16)

plt.axis('equal')
plt.tight_layout()
plt.savefig('results/pie.png', dpi=200, bbox_inches='tight')
plt.close()
