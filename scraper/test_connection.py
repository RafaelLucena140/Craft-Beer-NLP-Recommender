import requests

# URL do diretório de cervejas do BeerAdvocate
url = "https://www.beeradvocate.com/beer/"

# Header simulando um navegador real (Google Chrome no Windows)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print(f"Testando a conexão com: {url}")
resposta = requests.get(url, headers=headers)

print(f"Status Code retornado: {resposta.status_code}")

if resposta.status_code == 200:
    print("✅ Sucesso! O site permitiu o acesso (200 OK). O caminho está livre para o scraping.")
elif resposta.status_code in [403, 406]:
    print("❌ Bloqueado! O site identificou que somos um robô e negou o acesso.")
else:
    print("⚠️ Erro desconhecido ou página não encontrada.")