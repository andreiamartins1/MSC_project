import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Configurar a API do Kaggle
api = KaggleApi()
api.authenticate()

# Baixar o dataset de Cats vs Dogs
dataset_name = 'awsaf49/dogs-vs-cats'
download_path = './catsndogs_dataset.zip'

# Baixar o dataset do Kaggle
api.dataset_download_files(dataset_name, path=download_path, unzip=True)

# Caminho para os arquivos extraídos
extracted_folder = './dogs-vs-cats/'

# Criar diretórios de treinamento e validação
os.makedirs('data/catsndogs/train', exist_ok=True)
os.makedirs('data/catsndogs/val', exist_ok=True)

# Mover ou copiar imagens de acordo com a divisão entre treino e validação
import shutil
import random

# Lista todos os arquivos da pasta extraída
image_files = os.listdir(extracted_folder + 'train/')

# Embaralha a lista de imagens
random.shuffle(image_files)

# Define o número de imagens para treino (90%) e validação (10%)
split_index = int(0.9 * len(image_files))

# Move as imagens para as pastas correspondentes
for image_file in image_files[:split_index]:
    shutil.move(os.path.join(extracted_folder + 'train/', image_file), os.path.join('data/catsndogs/train', image_file))

for image_file in image_files[split_index:]:
    shutil.move(os.path.join(extracted_folder + 'train/', image_file), os.path.join('data/catsndogs/val', image_file))

print(f"Images organized into {len(image_files[:split_index])} for training and {len(image_files[split_index:])} for validation.")
