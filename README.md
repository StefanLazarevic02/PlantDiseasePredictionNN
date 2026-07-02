# Klasifikacija bolesti biljaka pomoću CNN i Transfer Learning pristupa

Ovaj repozitorijum sadrži implementaciju različitih arhitektura dubokih neuronskih mreža za višeklasnu klasifikaciju bolesti biljaka na osnovu slika listova (38 klasa). Projekat obuhvata ceo proces: od pripreme skupa i eliminacije curenja podataka, preko augmentacije, do poređenja performansi baznog i pretreniranih modela na validacionom i nezavisnom test skupu.

## Pregled projekta

Glavni fokus je na primeni Transfer Learning-a kako bi se postigla visoka tačnost na skupu New Plant Diseases Dataset (~87.867 slika, 14 biljnih vrsta, 26 bolesti i zdravi listovi). Implementirani modeli i komponente uključuju:

* **Baseline CNN:** Prilagođena arhitektura sa 3 konvoluciona sloja (trenirana od nule).
* **Transfer Learning modeli:** ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0 i VGG16.
* **Čišćenje podataka:** Eliminacija curenja podataka (data leakage) grupno-svesnom podelom zasnovanom na perceptualnom heširanju (phash/ahash/dhash + Union-Find).

## Tehnologije i zavisnosti

Za pokretanje koda neophodno je imati instaliran Python 3.10+ i sledeće biblioteke:

* **PyTorch & torchvision:** Osnovni framework za modele i pretrenirane arhitekture.
* **Albumentations & OpenCV:** Napredna augmentacija i učitavanje slika.
* **Scikit-learn:** Stratifikovana podela podataka i evaluacione metrike (F1, Precision, Recall, ROC-AUC).
* **ImageHash:** Perceptualno heširanje za detekciju duplikata i eliminaciju curenja podataka.
* **Pandas & NumPy:** Rad sa podacima.
* **Matplotlib & Seaborn:** Vizuelizacija rezultata i matrica konfuzije.

Instalacija:

```
pip install torch torchvision albumentations opencv-python scikit-learn imagehash pandas numpy matplotlib seaborn tqdm pyyaml kaggle
```

**Napomena:** U Google Colab okruženju je `Pillow` već instaliran, pa je za lokalno pokretanje potrebno i tu biblioteku instalirati i u tom slučaju instalacija izgleda ovako:
```
pip install torch torchvision albumentations opencv-python scikit-learn imagehash pandas numpy matplotlib seaborn tqdm pyyaml pillow kaggle
```

Ovaj projekat koristi duboko učenje (Deep Learning) za automatizovanu klasifikaciju slika listova u 38 kategorija. Fokus je na poređenju baznog CNN modela treniranog od nule sa pretreniranim (Transfer Learning) arhitekturama, uz posebnu pažnju na korektnu pripremu podataka radi realne procene performansi.

## Google Colab & GPU Ubrzanje

Projekat je primarno razvijen i optimizovan za rad u Google Colab okruženju. S obzirom na kompleksnost modela kao što su VGG16 i DenseNet-121, korišćenje grafičkog procesora (GPU) je neophodno za efikasno treniranje.

Kako podesiti Colab:

1. Otvorite `PlantDiseasePrediction.ipynb` fajl u Google Colab-u.
2. Idite na Runtime -> Change runtime type.
3. Pod Hardware accelerator izaberite T4 GPU (ili jači).
4. Pokrenite prvu ćeliju za montiranje Google Drive-a kako biste trajno čuvali istrenirane modele i rezultate:

```
from google.colab import drive
drive.mount('/content/drive')
```

5. Otpremite (upload) svoj `kaggle.json` fajl kada to odgovarajuća ćelija zatraži, kako biste omogućili preuzimanje skupa podataka direktno sa Kaggle platforme. Fajl mora biti u sledećem formatu:

```
{
  "username": "...",
  "key": "..."
}
```

Vrednosti `username` i `key` preuzimaju se sa Kaggle sajta: **Account → Settings → API → Create New Token**.
