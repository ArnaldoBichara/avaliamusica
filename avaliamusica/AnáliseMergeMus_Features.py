#%%
import pandas as pd
import numpy as np
import pickle
# 
dfUserAMusFeatures =  pd.read_pickle ("./arquivos intermediarios/UserAMus&Features.pickle")  
df600kMus_Features =  pd.read_pickle ("./arquivos intermediarios/600kMus&Features.pickle")  

# %%
dfUserAMusFeatures.tail(1)
#%%
df600kMus_Features.tail(1)

# %% ordenando o index. 
dfUserAMusFeatures.info()
df600kMus_Features.info()
#%%
print (dfUserAMusFeatures.columns.values)
print (df600kMus_Features.columns.values)
 
#%% buscando uma música específica da Enya nos dois datasets
print(dfUserAMusFeatures[dfUserAMusFeatures['musica'].str.contains("anywhere is", na= False, case=False)][['id_musica','artista', 'musica']].to_string(index=False))
print(df600kMus_Features[df600kMus_Features['musica'].str.contains("anywhere is", na= False, case=False)][['id_musica','artista', 'musica']].to_string(index=False))

#%% verificando quantas músicas de User A existe em 600kMus
listaMusEmComum = set(dfUserAMusFeatures['id_musica']).intersection(df600kMus_Features['id_musica'])
#%%
print (len(listaMusEmComum))
print (len (dfUserAMusFeatures))

#%%
len(df600kMus_Features)
# %% Incluindo músicas do user A no dataset de músicas & features
dfAudioFeatures = pd.concat([df600kMus_Features, dfUserAMusFeatures], ignore_index=True, verify_integrity=True)

# %%
len(dfAudioFeatures)

# %%
