import sys
import os
import hashlib
import torch
import clip
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QSlider, QProgressBar, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal


def best_device():
    """Pick the fastest available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ──────────────────────────────────────────────────────────────
# Per-image dict cache: { "path:mtime" -> embedding_tensor }
# Adding/removing/changing individual files only re-encodes those.
# ──────────────────────────────────────────────────────────────
class EmbeddingCache:
    def __init__(self, folder: str):
        h = hashlib.md5(folder.encode()).hexdigest()[:8]
        self._path = os.path.join(folder, f".clip_cache_{h}.pt")
        self._data: dict = {}   # key → 1-D cpu tensor
        self._load()

    def _file_key(self, path: str) -> str:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return f"{path}:{mtime}"

    def _load(self):
        if os.path.exists(self._path):
            try:
                self._data = torch.load(self._path, map_location="cpu")
                if not isinstance(self._data, dict):
                    self._data = {}
            except Exception:
                self._data = {}

    def save(self):
        try:
            torch.save(self._data, self._path)
        except Exception as e:
            print(f"[cache] save failed: {e}")

    def get(self, path: str):
        """Return cached tensor or None."""
        return self._data.get(self._file_key(path))

    def put(self, path: str, tensor):
        """Store a 1-D cpu tensor."""
        self._data[self._file_key(path)] = tensor

    def purge_missing(self, valid_paths: set):
        """Drop entries for files that no longer exist (keeps cache lean)."""
        dead = [k for k in self._data if not any(k.startswith(p) for p in valid_paths)]
        for k in dead:
            del self._data[k]


# ──────────────────────────────────────────────────────────────
# Background worker: indexes a folder using per-image cache
# ──────────────────────────────────────────────────────────────
class IndexWorker(QThread):
    progress  = pyqtSignal(int, int)   # (done, total)
    finished  = pyqtSignal(list)       # list of (path, tensor)
    new_count = pyqtSignal(int)        # how many were freshly encoded
    error     = pyqtSignal(str)

    def __init__(self, model, preprocess, device, folder, batch_size=64):
        super().__init__()
        self.model      = model
        self.preprocess = preprocess
        self.device     = device
        self.folder     = folder
        self.batch_size = batch_size

    def run(self):
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        all_paths = [
            os.path.join(self.folder, f)
            for f in os.listdir(self.folder)
            if os.path.splitext(f)[1].lower() in exts
        ]
        if not all_paths:
            self.finished.emit([])
            return

        cache = EmbeddingCache(self.folder)
        cache.purge_missing(set(all_paths))

        # Split into cached vs. needs encoding
        cached_pairs = []
        to_encode    = []
        for p in all_paths:
            emb = cache.get(p)
            if emb is not None:
                cached_pairs.append((p, emb))
            else:
                to_encode.append(p)

        total    = len(all_paths)
        done_so_far = len(cached_pairs)
        self.progress.emit(done_so_far, total)

        # Encode only new/changed images in batches
        new_pairs = []
        for batch_start in range(0, len(to_encode), self.batch_size):
            batch_paths = to_encode[batch_start:batch_start + self.batch_size]
            tensors, valid = [], []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self.preprocess(img))
                    valid.append(p)
                except Exception:
                    pass

            if tensors:
                batch_tensor = torch.stack(tensors).to(self.device)
                with torch.no_grad():
                    embs = self.model.encode_image(batch_tensor)
                    embs /= embs.norm(dim=-1, keepdim=True)
                embs = embs.cpu()

                for p, e in zip(valid, embs.unbind(0)):
                    cache.put(p, e)
                    new_pairs.append((p, e))

            done_so_far += len(batch_paths)
            self.progress.emit(done_so_far, total)

        if new_pairs:
            cache.save()

        all_pairs = cached_pairs + new_pairs
        self.new_count.emit(len(new_pairs))
        self.finished.emit(all_pairs)


# Image card
CARD_SIZE   = 200   # width and height of each grid card
THUMB_SIZE  = 160   # image area inside card


class ImageCard(QWidget):
    clicked = pyqtSignal(str)   # emits file path

    def __init__(self, path: str, similarity: float, parent=None):
        super().__init__(parent)
        self.path = path
        self.setFixedSize(CARD_SIZE, CARD_SIZE + 36)
        self.setCursor(Qt.PointingHandCursor)
        self._build(path, similarity)

    def _build(self, path, similarity):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 6)
        layout.setSpacing(4)

        # Thumbnail
        thumb_lbl = QLabel()
        thumb_lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb_lbl.setAlignment(Qt.AlignCenter)
        thumb_lbl.setStyleSheet(
            "background:#0d0d0d; border:1px solid #2a2a2a; border-radius:6px;"
        )
        try:
            img  = Image.open(path).convert("RGB")
            thumb = img.resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
            data  = thumb.tobytes()
            qimg  = QImage(data, THUMB_SIZE, THUMB_SIZE, THUMB_SIZE * 3, QImage.Format_RGB888)
            pix   = QPixmap.fromImage(qimg)
            thumb_lbl.setPixmap(pix.scaled(THUMB_SIZE, THUMB_SIZE,
                                           Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            thumb_lbl.setText("⚠")

        # Score badge + filename
        pct = similarity * 100
        if pct >= 50:   badge_color = "#5a5aff"
        elif pct >= 30: badge_color = "#aa5aff"
        else:           badge_color = "#ff5a8a"

        info_lbl = QLabel(
            f"<span style='color:{badge_color};font-weight:700'>{similarity:.2f}</span>"
            f" <span style='color:#888;font-size:11px'>{os.path.basename(path)}</span>"
        )
        info_lbl.setWordWrap(False)
        info_lbl.setStyleSheet("font-size:12px;")

        layout.addWidget(thumb_lbl, alignment=Qt.AlignHCenter)
        layout.addWidget(info_lbl)

    # hover highlight
    def enterEvent(self, _):
        self.setStyleSheet("background:#1f1f2e; border-radius:10px;")

    def leaveEvent(self, _):
        self.setStyleSheet("")

    def mousePressEvent(self, _):
        self.clicked.emit(self.path)


# Full image viewer dialog (opens on card click)
class ImageViewer(QWidget):
    def __init__(self, path: str, results: list, current_index: int, parent=None):
        super().__init__(parent, Qt.Window)
        self.results       = results          # list of (path, score)
        self.current_index = current_index
        self.setWindowTitle(os.path.basename(path))
        self.resize(900, 700)
        self.setStyleSheet("background:#0a0a0a; color:#eee;")
        self._build_ui()
        self._load(self.current_index)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # top bar
        bar = QWidget()
        bar.setFixedHeight(48)
        bar.setStyleSheet("background:#111; border-bottom:1px solid #222;")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(16, 0, 16, 0)

        self.title_lbl = QLabel()
        self.title_lbl.setStyleSheet("font-size:13px; color:#ccc;")

        self.score_lbl = QLabel()
        self.score_lbl.setStyleSheet("font-size:13px; font-weight:700; color:#5a5aff;")

        open_btn = QPushButton("Open in system viewer")
        open_btn.setStyleSheet(
            "background:#1e1e1e; border:1px solid #333; border-radius:5px;"
            "padding:4px 12px; color:#aaa; font-size:12px;"
        )
        open_btn.clicked.connect(self._open_externally)

        bar_layout.addWidget(self.title_lbl)
        bar_layout.addStretch()
        bar_layout.addWidget(self.score_lbl)
        bar_layout.addSpacing(20)
        bar_layout.addWidget(open_btn)
        root.addWidget(bar)

        # image area
        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(0)

        def nav_btn(symbol):
            b = QPushButton(symbol)
            b.setFixedWidth(52)
            b.setSizePolicy(b.sizePolicy().horizontalPolicy(),
                            b.sizePolicy().verticalPolicy())
            b.setStyleSheet(
                "background:#111; border:none; color:#555; font-size:26px;"
                "QPushButton:hover{color:#fff; background:#1e1e1e;}"
            )
            return b

        self.prev_btn = nav_btn("‹")
        self.next_btn = nav_btn("›")
        self.prev_btn.clicked.connect(lambda: self._navigate(-1))
        self.next_btn.clicked.connect(lambda: self._navigate(+1))

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background:#0a0a0a;")

        mid.addWidget(self.prev_btn)
        mid.addWidget(self.img_label, stretch=1)
        mid.addWidget(self.next_btn)
        root.addLayout(mid, stretch=1)

        # ── bottom: path ──
        self.path_lbl = QLabel()
        self.path_lbl.setStyleSheet(
            "background:#111; border-top:1px solid #222;"
            "padding:6px 16px; font-size:11px; color:#555;"
        )
        root.addWidget(self.path_lbl)

    def _load(self, idx: int):
        self.current_index = idx
        path, score = self.results[idx]
        self.setWindowTitle(os.path.basename(path))
        self.title_lbl.setText(f"{idx+1} / {len(self.results)}  ·  {os.path.basename(path)}")
        self.score_lbl.setText(f"score {score:.3f}")
        self.path_lbl.setText(path)
        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < len(self.results) - 1)

        try:
            pix = QPixmap(path)
            avail = self.img_label.size()
            if avail.width() < 10:  # not yet laid out
                avail = QSize(820, 580)
            self.img_label.setPixmap(
                pix.scaled(avail, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        except Exception:
            self.img_label.setText("Could not load image")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.results:
            self._load(self.current_index)

    def _navigate(self, delta: int):
        new_idx = self.current_index + delta
        if 0 <= new_idx < len(self.results):
            self._load(new_idx)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Right, Qt.Key_Space):
            self._navigate(+1)
        elif event.key() == Qt.Key_Left:
            self._navigate(-1)
        elif event.key() == Qt.Key_Escape:
            self.close()

    def _open_externally(self):
        import subprocess, platform
        path = self.results[self.current_index][0]
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])


# results flow grid
from PyQt5.QtWidgets import QScrollArea, QGridLayout


class ResultGrid(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(
            "QScrollArea { border:none; background:#111; }"
            "QScrollBar:vertical { background:#1a1a1a; width:8px; }"
            "QScrollBar::handle:vertical { background:#333; border-radius:4px; }"
        )
        self._container = QWidget()
        self._grid      = QGridLayout(self._container)
        self._grid.setSpacing(10)
        self._grid.setContentsMargins(12, 12, 12, 12)
        self.setWidget(self._container)
        self._results   = []   # (path, score)
        self._cols      = 4

    def set_results(self, results: list):
        """results = [(path, score), …] sorted best-first."""
        self._results = results
        self._repopulate()

    def _repopulate(self):
        # clear old cards
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cols = max(1, self.viewport().width() // (CARD_SIZE + 12))
        self._cols = cols

        for i, (path, score) in enumerate(self._results):
            card = ImageCard(path, score)
            card.clicked.connect(self._open_viewer)
            self._grid.addWidget(card, i // cols, i % cols)

        # push cards to top-left
        self._grid.setRowStretch(max(1, (len(self._results) - 1) // cols + 1), 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        new_cols = max(1, self.viewport().width() // (CARD_SIZE + 12))
        if new_cols != self._cols and self._results:
            self._repopulate()

    def _open_viewer(self, path: str):
        idx = next((i for i, (p, _) in enumerate(self._results) if p == path), 0)
        viewer = ImageViewer(path, self._results, idx, parent=self)
        viewer.show()


# Main window
class ClipSearcher(QWidget):
    def __init__(self):
        super().__init__()
        self.device = best_device()
        print(f"Loading CLIP model on {self.device}…")
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model.eval()

        self.folder_path = None
        self.index_pairs = []
        self.worker      = None

        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("CLIP Image Searcher")
        self.resize(1080, 740)
        self.setStyleSheet("""
            QWidget          { background:#111; color:#eee;
                               font-family:'Segoe UI',sans-serif; font-size:13px; }
            QLineEdit        { background:#1e1e1e; border:1px solid #333;
                               border-radius:6px; padding:6px 10px; color:#eee; }
            QPushButton      { background:#1e1e1e; border:1px solid #333;
                               border-radius:6px; padding:6px 14px; color:#ccc; }
            QPushButton:hover{ background:#2a2aee; border-color:#5a5aff; color:#fff; }
            QPushButton:disabled { background:#161616; color:#444; border-color:#222; }
            QSlider::groove:horizontal { height:4px; background:#2a2a2a; border-radius:2px; }
            QSlider::handle:horizontal { width:14px; height:14px; margin:-5px 0;
                                         background:#5a5aff; border-radius:7px; }
            QProgressBar     { background:#1a1a1a; border:1px solid #2a2a2a;
                               border-radius:4px; height:5px; }
            QProgressBar::chunk { background:#5a5aff; border-radius:4px; }
        """)

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(14, 14, 14, 14)

        # toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.folder_btn = QPushButton("📂 Folder")
        self.folder_btn.setFixedWidth(100)
        self.folder_btn.clicked.connect(self.choose_folder)

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("color:#555; font-size:12px;")

        self.prompt_field = QLineEdit()
        self.prompt_field.setPlaceholderText("Search prompt…  e.g.  'a dog playing outside'")
        self.prompt_field.returnPressed.connect(self.search_images)

        self.search_btn = QPushButton("🔍 Search")
        self.search_btn.setFixedWidth(100)
        self.search_btn.setEnabled(False)
        self.search_btn.clicked.connect(self.search_images)

        toolbar.addWidget(self.folder_btn)
        toolbar.addWidget(self.folder_label)
        toolbar.addWidget(self.prompt_field, stretch=1)
        toolbar.addWidget(self.search_btn)
        root.addLayout(toolbar)

        # threshold row
        thr_row = QHBoxLayout()
        thr_row.setSpacing(8)
        thr_row.addWidget(QLabel("Threshold"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(20)
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"{v/100:.2f}")
        )
        self.threshold_label = QLabel("0.20")
        self.threshold_label.setFixedWidth(34)
        thr_row.addWidget(self.threshold_slider, stretch=1)
        thr_row.addWidget(self.threshold_label)
        root.addLayout(thr_row)

        # progress and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(5)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#555; font-size:12px;")
        root.addWidget(self.status_label)

        self.grid = ResultGrid()
        root.addWidget(self.grid, stretch=1)

    # folder
    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self.folder_path = folder
        self.folder_label.setText(os.path.basename(folder))
        self.index_pairs = []
        self.search_btn.setEnabled(False)
        self.grid.set_results([])
        self.status_label.setText("Indexing…")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.folder_btn.setEnabled(False)

        self.worker = IndexWorker(self.model, self.preprocess, self.device, folder)
        self.worker.progress.connect(self._on_progress)
        self.worker.new_count.connect(lambda n: setattr(self, "_new_count", n))
        self.worker.finished.connect(self._on_indexed)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, done, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)
        self.status_label.setText(f"Indexing… {done} / {total}")

    def _on_indexed(self, pairs):
        self.index_pairs = pairs
        self.progress_bar.setVisible(False)
        self.folder_btn.setEnabled(True)
        self.search_btn.setEnabled(True)
        n_new = getattr(self, "_new_count", len(pairs))
        if n_new == 0:             detail = "all from cache"
        elif n_new == len(pairs):  detail = "freshly indexed"
        else:                      detail = f"{len(pairs)-n_new} cached · {n_new} new"
        self.status_label.setText(f"✅  {len(pairs)} images ready  ({detail})")

    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.folder_btn.setEnabled(True)
        self.status_label.setText(f"❌  {msg}")

    # search
    def search_images(self):
        prompt = self.prompt_field.text().strip()
        if not prompt or not self.index_pairs:
            return

        threshold = self.threshold_slider.value() / 100.0

        with torch.no_grad():
            text_emb = self.model.encode_text(clip.tokenize(prompt).to(self.device))
            text_emb /= text_emb.norm(dim=-1, keepdim=True)
            text_emb  = text_emb.cpu()

        all_embs = torch.stack([e for _, e in self.index_pairs])
        sims     = (all_embs @ text_emb.T).squeeze(1)

        results = [
            (self.index_pairs[i][0], sims[i].item())
            for i in range(len(self.index_pairs))
            if sims[i].item() >= threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        self.grid.set_results(results)
        self.status_label.setText(
            f"🔍  {len(results)} result{'s' if len(results)!=1 else ''} for {prompt}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ClipSearcher()
    win.show()
    sys.exit(app.exec_())