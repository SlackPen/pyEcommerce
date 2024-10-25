import os
import ast

def encontrar_imports(diretorio_raiz):
    """
    Percorre recursivamente todos os arquivos .py no diretório fornecido e encontra os pacotes importados.
    """
    imports = set()  # Usamos um set para evitar duplicatas

    for root, _, files in os.walk(diretorio_raiz):
        for file in files:
            if file.endswith('.py'):
                caminho_arquivo = os.path.join(root, file)
                with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                    try:
                        # Analisa a árvore de sintaxe do arquivo
                        tree = ast.parse(f.read(), filename=file)
                        for node in ast.walk(tree):
                            # Detecta importações do tipo: import x
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    imports.add(alias.name.split('.')[0])

                            # Detecta importações do tipo: from x import y
                            elif isinstance(node, ast.ImportFrom):
                                if node.module is not None:
                                    imports.add(node.module.split('.')[0])
                    except Exception as e:
                        print(f"Erro ao processar {caminho_arquivo}: {e}")

    return imports

# Caminho para o diretório do seu projeto
diretorio = "/caminho/para/seu/projeto"

# Executa a função e obtém os pacotes importados
pacotes_encontrados = encontrar_imports(diretorio)

# Exibe os pacotes encontrados
print("Pacotes encontrados:")
for pacote in pacotes_encontrados:
    print(pacote)

# Opcional: Salva os pacotes em um arquivo requirements.txt
with open('requirements.txt', 'w') as f:
    for pacote in sorted(pacotes_encontrados):
        f.write(f"{pacote}\n")



import importlib.util

for pacote in pacotes_encontrados:
    if importlib.util.find_spec(pacote) is None:
        print(f"Pacote não instalado: {pacote}")


import pkg_resources

with open('requirements.txt', 'w') as f:
    for pacote in sorted(pacotes_encontrados):
        try:
            versao = pkg_resources.get_distribution(pacote).version
            f.write(f"{pacote}=={versao}\n")
        except pkg_resources.DistributionNotFound:
            f.write(f"{pacote}\n")


