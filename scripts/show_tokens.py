# scripts/show_tokens.py
from transformers import AutoTokenizer

# It's a good practice to handle potential errors
try:
    # Load the tokenizer associated with our embedding model
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")

    sentence = "失智症可以治療嗎？"
    tokens = tokenizer.tokenize(sentence)

    print(f"原始句子: {sentence}")
    print(f"Tokens: {tokens}")

    # The tokenizer also converts tokens to their corresponding numerical IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Token IDs: {token_ids}")

    # You can also see the special tokens the model adds
    encoded_input = tokenizer(sentence)
    print(f"完整的編碼後輸入 (包含特殊 token): {encoded_input['input_ids']}")
    print(f"解碼回來的樣子: {tokenizer.decode(encoded_input['input_ids'])}")

except ImportError:
    print("錯誤：找不到 'transformers' 套件。")
    print("請執行 'uv sync --extra dev' 來安裝開發依賴。")
except Exception as e:
    print(f"發生錯誤：{e}")
    print("請確認模型 'BAAI/bge-large-zh-v1.5' 是否可以正常下載。")

