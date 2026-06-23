from search import build_bm25_index, hybrid_search

build_bm25_index()
results = hybrid_search('What NPT incidents happened during drilling?', category='npt_incidents')

print('Query: What NPT incidents happened during drilling?')
print('Category filter: npt_incidents')
print()
for i, r in enumerate(results, 1):
    print(f"Chunk {i}:")
    print(f"  filename  : {r['filename']}")
    print(f"  category  : {r['category']}")
    print(f"  heading   : {r['heading']}")
    print(f"  rrf_score : {r['rrf_score']}")
    print(f"  text      : {r['text'][:100]}...")
    print()