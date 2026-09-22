/**
 * Plagiarism Detector - Interactive Client Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  initWordCounters();
  initPresetLoader();
  initDropzones();
  initSyncScroll();
  initGaugeAnimation();
  initProgressBarAnimation();
  initFlashClose();
});

/* Animated Progress Bars */
function initProgressBarAnimation() {
  const fills = document.querySelectorAll('.progress-fill');
  fills.forEach(fill => {
    const targetWidth = fill.getAttribute('data-width') || fill.style.width;
    fill.style.width = '0%';
    setTimeout(() => {
      fill.style.width = targetWidth;
    }, 150);
  });
}

/* Flash message close buttons */
function initFlashClose() {
  document.querySelectorAll('.flash-close').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const alert = e.target.closest('.flash-alert');
      if (alert) {
        alert.style.opacity = '0';
        alert.style.transform = 'translateY(-10px)';
        setTimeout(() => alert.remove(), 250);
      }
    });
  });
}

/* Live Character & Word Counter */
function initWordCounters() {
  const doc1Area = document.getElementById('doc1_text');
  const doc2Area = document.getElementById('doc2_text');

  function updateCounts(textarea, wordPillId, charPillId) {
    if (!textarea) return;
    const text = textarea.value.trim();
    const words = text ? (text.match(/\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b/g) || []).length : 0;
    const chars = textarea.value.length;

    const wordEl = document.getElementById(wordPillId);
    const charEl = document.getElementById(charPillId);

    if (wordEl) wordEl.textContent = `${words} words`;
    if (charEl) charEl.textContent = `${chars} chars`;
  }

  if (doc1Area) {
    doc1Area.addEventListener('input', () => updateCounts(doc1Area, 'doc1_words_count', 'doc1_chars_count'));
    updateCounts(doc1Area, 'doc1_words_count', 'doc1_chars_count');
  }

  if (doc2Area) {
    doc2Area.addEventListener('input', () => updateCounts(doc2Area, 'doc2_words_count', 'doc2_chars_count'));
    updateCounts(doc2Area, 'doc2_words_count', 'doc2_chars_count');
  }

  // Clear buttons
  const clearBtn = document.getElementById('clear_all_btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', (e) => {
      e.preventDefault();
      if (doc1Area) doc1Area.value = '';
      if (doc2Area) doc2Area.value = '';
      updateCounts(doc1Area, 'doc1_words_count', 'doc1_chars_count');
      updateCounts(doc2Area, 'doc2_words_count', 'doc2_chars_count');
      document.querySelectorAll('.file-chosen-name').forEach(el => el.textContent = '');
    });
  }

  // Swap button
  const swapBtn = document.getElementById('swap_docs_btn');
  if (swapBtn) {
    swapBtn.addEventListener('click', (e) => {
      e.preventDefault();
      if (doc1Area && doc2Area) {
        const temp = doc1Area.value;
        doc1Area.value = doc2Area.value;
        doc2Area.value = temp;
        updateCounts(doc1Area, 'doc1_words_count', 'doc1_chars_count');
        updateCounts(doc2Area, 'doc2_words_count', 'doc2_chars_count');
      }
    });
  }
}

/* Sample Academic Presets for 1-Click QA Testing */
const PRESETS = {
  exact: {
    title: "Exact Duplicate (100%)",
    doc1: "Artificial intelligence and machine learning algorithms have demonstrated state-of-the-art capability in natural language processing and computer vision. By leveraging deep neural architectures trained on large-scale datasets, these models extract intricate statistical patterns without manual feature engineering.",
    doc2: "Artificial intelligence and machine learning algorithms have demonstrated state-of-the-art capability in natural language processing and computer vision. By leveraging deep neural architectures trained on large-scale datasets, these models extract intricate statistical patterns without manual feature engineering."
  },
  paraphrased: {
    title: "Paraphrased Text (~60-80%)",
    doc1: "Renewable energy sources such as solar and wind power are critical for mitigating global climate change. Transitioning away from fossil fuels reduces greenhouse gas emissions and fosters long-term environmental sustainability across industrialized nations.",
    doc2: "Clean renewable power options like solar panels and wind turbines play an essential role in combating planetary climate warming. Shifting away from coal and oil lowers carbon gas emissions and supports enduring environmental stability in modern countries."
  },
  transposed: {
    title: "Transposed Sentences (Swapped Paragraphs)",
    doc1: "Deep learning models require extensive computational resources during backpropagation. Transformers have largely replaced recurrent neural networks for sequence modeling tasks due to their self-attention mechanism.",
    doc2: "Transformers have largely replaced recurrent neural networks for sequence modeling tasks due to their self-attention mechanism. Deep learning models require extensive computational resources during backpropagation."
  },
  different: {
    title: "Completely Different (0%)",
    doc1: "Quantum computing utilizes the principles of superposition and quantum entanglement to execute complex calculations exponentially faster than classical Turing architectures.",
    doc2: "The Renaissance era in Western Europe sparked a profound cultural rebirth in classical philosophy, humanistic literature, figurative painting, and architectural aesthetics."
  }
};

function initPresetLoader() {
  const chips = document.querySelectorAll('.preset-chip');
  const doc1Area = document.getElementById('doc1_text');
  const doc2Area = document.getElementById('doc2_text');

  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      const presetKey = chip.getAttribute('data-preset');
      const preset = PRESETS[presetKey];
      if (preset && doc1Area && doc2Area) {
        doc1Area.value = preset.doc1;
        doc2Area.value = preset.doc2;
        doc1Area.dispatchEvent(new Event('input'));
        doc2Area.dispatchEvent(new Event('input'));
      }
    });
  });
}

/* Drag-and-Drop File Upload Reader (with Image OCR Support) */
const IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'webp', 'gif'];

function isImageFile(file) {
  if (file.type && file.type.startsWith('image/')) return true;
  const ext = (file.name || '').split('.').pop().toLowerCase();
  return IMAGE_EXTENSIONS.includes(ext);
}

function initDropzones() {
  setupDropzone('doc1_dropzone', 'doc1_file', 'doc1_text', 'doc1_filename', 'doc1_ocr_status');
  setupDropzone('doc2_dropzone', 'doc2_file', 'doc2_text', 'doc2_filename', 'doc2_ocr_status');
}

function setupDropzone(dropzoneId, inputId, textareaId, filenameId, ocrStatusId) {
  const dropzone = document.getElementById(dropzoneId);
  const input = document.getElementById(inputId);
  const textarea = document.getElementById(textareaId);
  const filenameDisplay = document.getElementById(filenameId);
  const ocrStatus = document.getElementById(ocrStatusId);

  if (!dropzone || !input) return;

  dropzone.addEventListener('click', () => input.click());

  input.addEventListener('change', () => {
    if (input.files.length > 0) {
      handleFile(input.files[0], textarea, filenameDisplay, ocrStatus);
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      input.files = files;
      handleFile(files[0], textarea, filenameDisplay, ocrStatus);
    }
  });
}

function handleFile(file, textarea, filenameDisplay, ocrStatus) {
  if (filenameDisplay) {
    filenameDisplay.textContent = `Selected: ${file.name} (${Math.round(file.size / 1024)} KB)`;
  }

  if (isImageFile(file)) {
    // Run client-side OCR using Tesseract.js
    handleImageOCR(file, textarea, ocrStatus);
  } else {
    // Read as plain text
    if (ocrStatus) ocrStatus.textContent = '';
    const reader = new FileReader();
    reader.onload = (e) => {
      if (textarea) {
        textarea.value = e.target.result;
        textarea.dispatchEvent(new Event('input'));
      }
    };
    reader.readAsText(file);
  }
}

async function handleImageOCR(file, textarea, ocrStatus) {
  if (typeof Tesseract === 'undefined') {
    if (ocrStatus) {
      ocrStatus.textContent = '⚠ OCR engine not loaded. Please reload the page.';
      ocrStatus.className = 'ocr-status ocr-error';
    }
    return;
  }

  if (ocrStatus) {
    ocrStatus.innerHTML = '<span class="ocr-spinner"></span> Extracting text from image… Please wait.';
    ocrStatus.className = 'ocr-status ocr-loading';
  }

  try {
    const result = await Tesseract.recognize(file, 'eng', {
      logger: (m) => {
        if (m.status === 'recognizing text' && ocrStatus) {
          const pct = Math.round((m.progress || 0) * 100);
          ocrStatus.innerHTML = `<span class="ocr-spinner"></span> OCR Processing… ${pct}%`;
        }
      }
    });

    const extractedText = (result.data.text || '').trim();

    if (textarea) {
      textarea.value = extractedText;
      textarea.dispatchEvent(new Event('input'));
    }

    if (ocrStatus) {
      if (extractedText.length > 0) {
        const wordCount = (extractedText.match(/\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b/g) || []).length;
        ocrStatus.textContent = `✓ OCR complete — extracted ${wordCount} words from image.`;
        ocrStatus.className = 'ocr-status ocr-success';
      } else {
        ocrStatus.textContent = '⚠ OCR could not extract any text. Try a clearer image.';
        ocrStatus.className = 'ocr-status ocr-error';
      }
    }
  } catch (err) {
    console.error('OCR Error:', err);
    if (ocrStatus) {
      ocrStatus.textContent = '✕ OCR failed: ' + (err.message || 'Unknown error');
      ocrStatus.className = 'ocr-status ocr-error';
    }
  }
}

/* Synchronized Scroll for Side-by-Side Comparison */
function initSyncScroll() {
  const pane1 = document.getElementById('comparison_pane_1');
  const pane2 = document.getElementById('comparison_pane_2');
  const toggle = document.getElementById('sync_scroll_toggle');

  if (!pane1 || !pane2) return;

  let isSyncing = true;
  let activePane = null;

  if (toggle) {
    toggle.addEventListener('change', (e) => {
      isSyncing = e.target.checked;
    });
  }

  function sync(source, target) {
    if (!isSyncing || activePane !== source) return;
    const percentage = source.scrollTop / (source.scrollHeight - source.clientHeight);
    target.scrollTop = percentage * (target.scrollHeight - target.clientHeight);
  }

  pane1.addEventListener('mouseenter', () => activePane = pane1);
  pane2.addEventListener('mouseenter', () => activePane = pane2);

  pane1.addEventListener('scroll', () => sync(pane1, pane2));
  pane2.addEventListener('scroll', () => sync(pane2, pane1));
}

/* Animated SVG Circular Gauge */
function initGaugeAnimation() {
  const gauges = document.querySelectorAll('.circle-meter');
  gauges.forEach(gauge => {
    const percent = parseFloat(gauge.getAttribute('data-percentage') || 0);
    const circle = gauge.querySelector('.meter-fill');
    if (circle) {
      const radius = circle.r.baseVal.value;
      const circumference = 2 * Math.PI * radius;
      circle.style.strokeDasharray = `${circumference} ${circumference}`;
      const offset = circumference - (percent / 100) * circumference;
      // Animate from empty
      circle.style.strokeDashoffset = circumference;
      setTimeout(() => {
        circle.style.strokeDashoffset = offset;
      }, 150);
    }
  });
}
