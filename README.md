# 🇹🇷 Türkçe NER (Varlık İsmi Tanıma) Görevinde BERT ve XLM-RoBERTa Karşılaştırması

Bu proje, Türkçe metinler üzerinde Varlık İsmi Tanıma (Named Entity Recognition - NER) görevi için iki farklı dil modelinin (monolingual ve multilingual) ince ayar (fine-tuning) süreçlerini, tokenizasyon stratejilerini ve performanslarını karşılaştırmalı olarak analiz etmektedir.

## 🎯 Projenin Amacı
Amacımız, sadece Türkçe derlemlerle eğitilmiş **BERTurk** modeli ile 100 farklı dilde eğitilmiş çok dilli **XLM-RoBERTa** modelinin NER görevindeki (Kişi, Kurum, Lokasyon tespiti) yeteneklerini ölçmek ve model mimarilerinin (WordPiece vs. SentencePiece) tokenizasyon farklılıklarının etiket doğruluğuna etkisini gözlemlemektir.

## 📊 Veri Seti
Çalışmada, çok dilli Vikipedi metinlerinden oluşan **`wikiann` (tr)** veri seti kullanılmıştır.
*   **Etiket Formatı:** BIO formatı (B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O)
*   **Veri Dağılımı:**
    *   Eğitim (Train): 20.000 cümle
    *   Doğrulama (Validation): 8.735 cümle
    *   Test: 8.694 cümle (Veri sızıntısını önlemek için eğitim setiyle örtüşen cümleler temizlenmiştir).

## 🧠 Kullanılan Modeller ve Metodoloji
1.  **`dbmdz/bert-base-turkish-cased` (BERTurk):** ~35 GB Türkçe veri ile eğitilmiş, WordPiece algoritması kullanan yerel uzman model.
2.  **`xlm-roberta-base` (XLM-R):** ~2.5 TB CommonCrawl verisi ile eğitilmiş, SentencePiece algoritması kullanan çok dilli model.

**Kritik Ön İşleme Adımları:**
*   **Alt-Sözcük ve Etiket Hizalaması (Label Alignment):** Modellerin kelimeleri parçalamasından kaynaklanan boyut uyuşmazlığını önlemek için, gerçek NER etiketleri sadece kelimenin ilk alt-sözcüğüne (subword) atanmış, devam eden eklere ve özel tokenlara (`[CLS]`, `[SEP]`, `<s>`) kayıp (loss) hesaplamasında yoksayılmaları için `-100` değeri atanmıştır.
*   **Katman Bazlı Öğrenme Oranı (LLRD):** XLM-R eğitiminde, modelin ön eğitimle kazandığı dilbilgisi yeteneklerini unutmaması için alt katmanlara düşük, sınıflandırıcı üst katmanlara yüksek öğrenme oranı uygulanmıştır.

## 🏆 Sonuçlar ve Performans (Test Seti)
Her iki model de doğruluk (accuracy) yanılgısına düşmemek adına **F1-Skoru (seqeval)** metrikleri ile değerlendirilmiştir.

| Kategori | BERTurk F1-Skoru | XLM-RoBERTa F1-Skoru |
| :--- | :---: | :---: |
| **LOC (Lokasyon)** | 0.916 | 0.915 |
| **ORG (Kurum)** | 0.868 | 0.876 |
| **PER (Kişi)** | 0.937 | 0.936 |
| **Genel Ortalama (Overall)** | **0.909** | **0.911** |

### 🔍 Hata Analizi ve Temel Çıkarımlar
*   **Tokenizasyon Etkisi:** Bir kelime ne kadar çok alt-parçaya (subword) bölünürse, modellerin o kelimenin varlık sınıfını doğru tahmin etme oranı düşmektedir.
*   **Kültürel ve Yabancı İsimler:** BERTurk, Türkçe eklere ve yerel coğrafi isimlere (örn: *Klazomenai*, *Perşembe*) daha iyi hakimken; XLM-RoBERTa çok dilli yapısı sayesinde yabancı kişi isimlerini (örn: *LL Cool J*, *Schwarzenegger*) ve birleşik kurum isimlerini (örn: *Nobel Edebiyat Ödülü*) çok daha bütüncül ve doğru kavramıştır.

## 💻 Kurulum ve Çalıştırma
Bu projeyi kendi ortamınızda çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

```bash
pip install transformers datasets evaluate seqeval torch pandas matplotlib
