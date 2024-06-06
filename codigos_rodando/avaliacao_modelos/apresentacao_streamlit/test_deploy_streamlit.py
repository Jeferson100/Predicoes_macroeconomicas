from selenium import webdriver

# Configurar o WebDriver (neste exemplo, usando o Chrome)
driver = webdriver.Chrome()

url = "https://predicao-selic.streamlit.app/"

driver.get(url)
if "Streamlit" in driver.title:  # Ajuste esta verificação conforme necessário
    print(f"O link do app Streamlit: {url} esta ativo.")