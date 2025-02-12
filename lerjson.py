import pandas as pd, glob, json
pd.concat([pd.DataFrame(json.load(open(f, 'r', encoding='utf-8'))['mensagens_violacao']) for f in glob.glob('*.json')], ignore_index=True).to_pickle('mensagens_violacao.pickle')
