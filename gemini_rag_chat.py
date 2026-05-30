import os
import numpy as np
from google import genai
from dotenv import load_dotenv

# Pastikan Anda telah menginstal dependensi:
# pip3 install google-genai numpy python-dotenv

def cosine_similarity(a, b):
    """Menghitung cosine similarity antara dua vektor."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def load_and_chunk_document(file_path):
    """Membaca dokumen dan memecahnya menjadi paragraf (chunks)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Memecah berdasarkan baris kosong (paragraf)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs

def get_embeddings(client, texts):
    """Mendapatkan embeddings untuk daftar teks menggunakan model Gemini."""
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents=texts,
    )
    # Mengambil nilai vektor (values) dari setiap embedding
    return [emb.values for emb in response.embeddings]

def main():
    # Load environment variables dari file .env
    load_dotenv()

    # Pengecekan API Key
    if "GEMINI_API_KEY" not in os.environ or os.environ["GEMINI_API_KEY"] == "api_key_anda_di_sini":
        print("Error: Variabel environment GEMINI_API_KEY belum diset dengan benar.")
        print("Silakan buka file .env dan isi GEMINI_API_KEY dengan API key asli Anda.")
        return

    # Inisialisasi Google GenAI client (akan secara otomatis mengambil dari environment GEMINI_API_KEY)
    client = genai.Client()

    print("Menginisialisasi RAG (Retrieval-Augmented Generation)...")
    
    # 1. Menyiapkan Dokumen Konteks
    doc_path = "document.txt"
    if not os.path.exists(doc_path):
        print(f"Error: File {doc_path} tidak ditemukan.")
        print("Harap buat file 'document.txt' terlebih dahulu di folder yang sama.")
        return
        
    chunks = load_and_chunk_document(doc_path)
    print(f"Berhasil memuat dokumen. Terdapat {len(chunks)} bagian (chunks).")
    
    print("Membuat embeddings untuk isi dokumen... (Harap tunggu)")
    chunk_embeddings = get_embeddings(client, chunks)
    print("Embeddings siap!\n")

    # 2. Menentukan System Instruction
    system_instruction = (
        "Anda adalah asisten AI dari perusahaan DDI (Data Dynamics Indonesia) yang ramah dan cerdas. "
        "Anda sedang berinteraksi dalam sebuah percakapan, jadi ingatlah selalu identitas pengguna dan histori chat sebelumnya. "
        "Pada setiap pesan pengguna, sistem mungkin akan menyertakan 'Konteks Tambahan' dari dokumen. "
        "Gunakan konteks tambahan tersebut HANYA JIKA relevan untuk menjawab pertanyaan tentang DDI. "
        "Jika pengguna menanyakan hal di luar konteks dokumen (misalnya tentang diri mereka, atau obrolan santai), "
        "jawablah secara natural berdasarkan histori percakapan atau pengetahuan umum Anda, tanpa perlu menyebutkan "
        "bahwa itu di luar dokumen resmi (kecuali jika benar-benar ditanya tentang fakta spesifik perusahaan)."
    )

    # Inisialisasi sesi Gemini Chat
    chat = client.chats.create(
        model="gemini-3.5-flash",
        config={
            "system_instruction": system_instruction,
            "temperature": 0.5
        }
    )

    print("="*60)
    print("🤖 Chatbot Gemini dengan RAG Siap!")
    print("Ketik 'keluar', 'exit', atau 'quit' untuk menghentikan program.")
    print("="*60)

    # 3. Loop Interaktif di Terminal untuk Tanya Jawab
    while True:
        try:
            user_input = input("\nAnda: ")
            
            if user_input.lower() in ['keluar', 'exit', 'quit']:
                print("Chatbot dimatikan. Sampai jumpa!")
                break
                
            if not user_input.strip():
                continue

            # --- LANGKAH RAG (Retrieval) ---
            # Embed pertanyaan pengguna
            query_embed_response = client.models.embed_content(
                model='gemini-embedding-2',
                contents=user_input,
            )
            query_embedding = query_embed_response.embeddings[0].values

            # Hitung kesamaan (similarity) kosinus antara pertanyaan dan semua chunk dokumen
            similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in chunk_embeddings]
            
            # Ambil maksimal 2 chunk paling relevan
            top_k = 2
            # np.argsort mengurutkan dari kecil ke besar, jadi kita ambil elemen terakhir dan balik urutannya
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Gabungkan chunk yang relevan sebagai konteks (misal dengan threshold similarity > 0.4)
            retrieved_chunks = [chunks[i] for i in top_indices if similarities[i] > 0.4]
            retrieved_context = "\n\n".join(retrieved_chunks)

            # --- LANGKAH AUGMENTASI PROMPT ---
            if retrieved_context:
                augmented_prompt = (
                    f"[Konteks Tambahan (hanya gunakan jika relevan dengan pertanyaan): {retrieved_context}]\n\n"
                    f"{user_input}"
                )
                print(f"[Info RAG: Menemukan {len(retrieved_chunks)} bagian dokumen yang mungkin relevan]")
            else:
                augmented_prompt = user_input
                print("[Info RAG: Tidak ada dokumen spesifik yang disisipkan untuk pertanyaan ini]")

            # --- LANGKAH GENERATION ---
            # Kirim pesan (yang sudah diaugmentasi konteks) ke Gemini Chat
            response = chat.send_message(augmented_prompt)
            print(f"Gemini: {response.text}")
            
        except KeyboardInterrupt:
            print("\nChatbot dimatikan paksa. Sampai jumpa!")
            break
        except Exception as e:
            print(f"Terjadi kesalahan saat memproses permintaan: {e}")

if __name__ == "__main__":
    main()
