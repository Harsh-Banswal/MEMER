#  MEMER

> **Real-Time Pose-to-Meme Similarity Matching Engine**  
> Powered by MediaPipe, FAISS (Facebook AI Similarity Search), and FastAPI.

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.111-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-v0.10-blue?logo=google&logoColor=white)](https://google.github.io/mediapipe/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU_1.8-orange?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-purple)](https://opensource.org/licenses/MIT)

**MEMER** is a real-time web application that matches your physical body poses and hand gestures to a database of **20,000+ popular internet memes** in under **0.1 milliseconds**. The project leverages a high-tech glowing skeleton overlay natively rendered in your browser and utilizes server-side parallel processing to perform blazing-fast vector similarity searches.

---

## Key Features

* Real-Time Pose Estimation:** Uses Google MediaPipe on the backend with hardware-accelerated EGL rendering to extract 106-dimensional pose vectors (upper body, shoulders, elbows, wrists, and hand joints) at **35+ FPS**.
*  Distance & Scale Invariant Matching:** Pose normalization algorithms ensure matching remains 100% accurate regardless of your distance from the camera or coordinate scale.
*  Collision-Free Dataset Merging:** Automatically handles sequential ZIP archive imports, prepending parent-directory structures and zip stems to guarantee no image filename collisions.
*  SIMD Vector Acceleration:** Employs **FAISS** (FlatIP index) with L2-normalized cosine similarity for sub-millisecond search execution.

---

##  Installation & Setup

Follow these steps to set up **MEMER** on your local machine:

### 1. Clone & Navigate to Project
```bash
cd /home/harsh/Desktop/POSEMEME/memer
```

### 2. Set Up the Virtual Environment
Create a clean isolated virtual environment and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Libraries
Upgrade pip and install all required system libraries listed in `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

##  Datasets & Merging

To achieve maximum matching coverage, **MEMER** supports merging multiple massive meme libraries seamlessly:

* **Dataset 1: Reddit Memes Dataset (Zip)**  
  🔗 [Kaggle Link](https://kaggle.com/datasets/sayangoswami/reddit-memes-dataset)
* **Dataset 2: Memotion Dataset 7k (Zip)**  
  🔗 [Kaggle Link](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k)

### How to Merge Datasets Collision-Free:

To combine both datasets into a single, unified search space without any filename collisions or overlaps, run the following commands sequentially:

```bash
# 1. Clear any old metadata indexes
rm data/metadata.json

# 2. Extract and register the Reddit Memes Dataset ZIP
python scripts/download_memes.py "filename1.zip"

# 3. Extract and cleanly append the Memotion Dataset ZIP
python scripts/download_memes.py "filename2.zip"
```

and many more dataset files as you download. By default, the script will create a folder with the name of the zip file and extract the contents of the zip file into that folder. It will also create a metadata file that will be used to store the path to the extracted files. Once all the datasets are extracted and registered, you can build the FAISS index database.

Once merged, your `data/metadata.json` will safely hold **20,000+ unique templates** with isolated prefixes (e.g. `local_filename1_...` and `local_filename2_...`).


For Improving the efficiency of the meme matching, You can add more dataset files for building FAISS Database. 

---

##  Building the FAISS Index Database

Once your datasets are merged, build your high-performance parallel FAISS vector database:

```bash
python scripts/build_index.py
```

### Performance Optimizations Active:
* **GPU/EGL Acceleration:** MediaPipe uses EGL graphics pipelines to utilize your graphics card for extraction.
* **Cap Workers Capping:** Automatic threading is safely capped at a maximum of **3 workers** to prevent system Out-Of-Memory (OOM) crashes while running 300% faster.
* **Density Filter:** The builder automatically filters out text-only or animal-only memes, keeping your index dense and accurate (**7,400+ valid pose vectors**).

---

##  Running the Web Application

Start the FastAPI application server:

```bash
.venv/bin/uvicorn app.main:app --reload
```

Open your browser and navigate to **[http://localhost:8000](http://localhost:8000)**. 
* Click **"Start Webcam"** and step back so your upper body is clearly visible in the frame!

---

##  Improving Matching Efficiency & Accuracy

To get the absolute best, most accurate matches for your poses:

1. **Upper Body Visibility:** Keep both shoulders fully visible in the webcam frame. The system uses the distance between your left and right shoulder as the baseline origin and scale.
2. **Mimic Head Posture:** Although the system ignores simple facial expressions (like smiling/winking) to maintain high performance, it **fully tracks head tilt, height, and head rotation**. 
   * If a meme character is looking down in sadness, tilt your head down!
   * If they are tilting their head in confusion, tilt your head sideways!
3. **Bold Hand Gestures:** Raise your arms, point to the sky, or cross your arms to match iconic gestures perfectly!

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
