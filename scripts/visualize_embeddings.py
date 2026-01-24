"""
This script generates a 2D visualization of sentence embeddings
to illustrate the concept of semantic similarity in a vector space.

It performs the following steps:
1. Loads the HuggingFace embedding model (BAAI/bge-large-zh-v1.5).
2. Defines a list of sentences with varying degrees of similarity.
3. Computes the embeddings (vectors) for each sentence.
4. Uses PCA to reduce the high-dimensional vectors to 2D.
5. Creates a scatter plot of the 2D vectors using Matplotlib.
6. Annotates each point with its corresponding sentence.
7. Saves the resulting plot to the 'docs/' directory.
"""
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

def get_chinese_font():
    """
    Finds and returns a Chinese font available on the system.
    This is necessary for Matplotlib to render Chinese characters correctly.
    """
    # Common Chinese fonts, add more if needed
    font_names = [
        "Microsoft YaHei", "SimHei", "Heiti TC", "PingFang SC",
        "WenQuanYi Zen Hei", "AR PL UKai CN"
    ]
    for font_name in font_names:
        try:
            # Check if the font is available
            fm.findfont(font_name, fallback_to_default=False)
            return font_name
        except Exception:
            continue
    
    # If no specific font is found, try to find any available CJK font
    for font in fm.fontManager.ttflist:
        if "cjk" in font.name.lower() or "chinese" in font.name.lower():
            return font.name
            
    print("警告：找不到可用的中文字體，圖表中的中文可能無法正常顯示。")
    print("請嘗試安裝 'wqy-zenhei' 或 'noto-fonts-cjk' 等字體套件。")
    return None

def visualize_embeddings():
    """
    Generates and saves a 2D plot of sentence embeddings.
    """
    print("載入 Embedding 模型 (BAAI/bge-large-zh-v1.5)...")
    # 1. Load the embedding model
    model_name = "BAAI/bge-large-zh-v1.5"
    model_kwargs = {"device": "cpu"} # Use "cuda" if GPU is available
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # 2. Define sentences
    sentences = [
        "失智症藥物有哪些副作用？",           # Query 1
        "服用失智症的藥會不會不舒服？",        # Query 2 (Similar to 1)
        "失智症可以治療嗎？",                 # Query 3 (Related but different)
        "今天天氣如何？",                     # Query 4 (Unrelated)
    ]
    colors = ['r', 'r', 'b', 'g'] # Assign colors to sentences

    print("計算句子向量...")
    # 3. Compute embeddings
    vectors = embeddings.embed_documents(sentences)
    vectors_np = np.array(vectors)

    print("使用 PCA 進行降維...")
    # 4. Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors_np)

    print("產生視覺化圖表...")
    # 5. Create plot
    plt.figure(figsize=(12, 8))

    # Set Chinese font
    font_name = get_chinese_font()
    if font_name:
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False # Fix for minus sign
    
    # Scatter plot
    for i, (sentence, (x, y)) in enumerate(zip(sentences, vectors_2d)):
        plt.scatter(x, y, c=colors[i], s=100)
        plt.text(x + 0.01, y + 0.01, sentence, fontsize=12, ha='left')

    plt.title("句子向量在二維空間中的分佈 (PCA 降維)", fontsize=16)
    plt.xlabel("第一主成分 (Principal Component 1)", fontsize=12)
    plt.ylabel("第二主成分 (Principal Component 2)", fontsize=12)
    plt.grid(True)
    
    # Add annotations
    plt.text(0.5, -0.1, 
             "語意相似的句子（紅色）在向量空間中距離較近", 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='darkred')

    # 7. Save the plot
    output_path = Path("docs/embedding_visualization.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    
    print(f"圖表已儲存至: {output_path}")

if __name__ == "__main__":
    visualize_embeddings()
