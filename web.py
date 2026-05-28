from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, send_file   # ADDED send_file
import numpy as np
from scipy import signal
from scipy.io import wavfile
import tempfile
import sympy as sp
from sympy import symbols, together
import json
import os
import uuid
from datetime import datetime
from PIL import Image, ImageFilter, ImageOps
import io
import base64
import cv2
import numpy as np
from PIL import Image
import io
import base64
import uuid

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'm4a', 'ogg', 'flac', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

PROJECTS_FILE = 'projects.json'

def load_projects():
    if os.path.exists(PROJECTS_FILE):
        with open(PROJECTS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_projects(projects):
    with open(PROJECTS_FILE, 'w') as f:
        json.dump(projects, f, indent=2)

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    """Upload an image file and return base64 preview"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Open image and convert to RGB
        img = Image.open(file.stream).convert('RGB')
        
        # Resize if too large (max 800px width/height)
        max_size = 800
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save to bytes and encode as base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Store original image in session or temp file
        if not hasattr(app, 'temp_images'):
            app.temp_images = {}
        import uuid
        img_id = str(uuid.uuid4())[:8]
        app.temp_images[img_id] = img
        
        return jsonify({
            'image_id': img_id,
            'original': f'data:image/png;base64,{img_base64}',
            'width': img.size[0],
            'height': img.size[1]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_image', methods=['POST'])
def process_image():
    """Apply selected filter to the image"""
    data = request.get_json()
    img_id = data.get('image_id')
    filter_type = data.get('filter_type', 'gaussian')
    intensity = data.get('intensity', 2.0)
    threshold = data.get('threshold', 50)  # Not used by new filters, but kept for compatibility

    if not hasattr(app, 'temp_images') or img_id not in app.temp_images:
        return jsonify({'error': 'Image not found'}), 404

    # Get the PIL Image object from our temporary storage
    img = app.temp_images[img_id]
    
    # Apply the filter
    processed_img = apply_image_filter(img, filter_type, intensity)

    # Convert the processed OpenCV image back to a PIL Image
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    processed_pil_img = Image.fromarray(processed_img_rgb)

    # Save to bytes and encode as base64
    buffered = io.BytesIO()
    processed_pil_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        'processed': f'data:image/png;base64,{img_base64}'
    })

# ---------- Simple EQ filters (biquad) ----------
# Replace your EQ functions with these more stable versions:

def apply_low_shelf(sig, fs, gain_db, cutoff=100):
    """Stable low-shelf filter for bass boost/cut"""
    if gain_db == 0:
        return sig
    
    # Use scipy's built-in filter design for better stability
    from scipy.signal import butter, lfilter
    
    # Convert gain to linear
    gain_linear = 10 ** (gain_db / 20.0)
    
    # Design a simple low-shelf using a 1st order filter
    # Normalize frequency
    wc = cutoff / (fs / 2)
    
    # Simple low-shelf implementation
    if gain_db > 0:  # Boost
        # Boost using peaking EQ
        Q = 0.707
        w0 = 2 * np.pi * cutoff / fs
        A = 10 ** (gain_db / 40.0)
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * Q)
        
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    else:  # Cut
        w0 = 2 * np.pi * cutoff / fs
        A = 10 ** (-gain_db / 40.0)
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * 0.707)
        
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    
    b = [b0/a0, b1/a0, b2/a0]
    a = [1.0, a1/a0, a2/a0]
    
    return lfilter(b, a, sig)

def apply_peak_eq(sig, fs, gain_db, freq=1000, Q=1.0):
    """Stable peaking filter for mid frequencies"""
    if gain_db == 0:
        return sig
    
    from scipy.signal import lfilter
    
    w0 = 2 * np.pi * freq / fs
    A = 10 ** (gain_db / 40.0)
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * Q)
    
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A
    
    b = [b0/a0, b1/a0, b2/a0]
    a = [1.0, a1/a0, a2/a0]
    
    return lfilter(b, a, sig)

def apply_high_shelf(sig, fs, gain_db, cutoff=5000):
    """Stable high-shelf filter for treble boost/cut"""
    if gain_db == 0:
        return sig
    
    from scipy.signal import lfilter
    
    w0 = 2 * np.pi * cutoff / fs
    A = 10 ** (gain_db / 40.0)
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * 0.707)
    
    if gain_db > 0:  # Boost
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    else:  # Cut
        A = 10 ** (-gain_db / 40.0)
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    
    b = [b0/a0, b1/a0, b2/a0]
    a = [1.0, a1/a0, a2/a0]
    
    return lfilter(b, a, sig)

# ---------- DSP Helper Functions (unchanged) ----------
def generate_signal(fs, duration, freq, amp, phase_deg, waveform):
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    phase_rad = np.deg2rad(phase_deg)
    if waveform == 'sine':
        sig = amp * np.sin(2*np.pi*freq*t + phase_rad)
    elif waveform == 'cosine':
        sig = amp * np.cos(2*np.pi*freq*t + phase_rad)
    else:
        sig = amp * np.random.normal(0, 0.1, len(t))
    return t.tolist(), sig.tolist()

def apply_filter(sig, fs, filter_type, cutoff1, cutoff2, order):
    """Apply filter without oscillation - using filtfilt for stability"""
    sig = np.array(sig, dtype=np.float64)
    nyq = fs / 2.0
    
    if nyq <= 0 or len(sig) == 0:
        return sig.tolist()
    
    # Force order to 1 for absolute stability
    order = 1
    
    try:
        if filter_type == 'lowpass':
            cutoff = min(cutoff1, nyq * 0.9)
            normal = cutoff / nyq
            normal = max(0.05, min(0.95, normal))
            print(f"Lowpass: {cutoff}Hz, norm={normal:.4f}")
            b, a = signal.butter(order, normal, btype='low')
            filtered = signal.filtfilt(b, a, sig)
            
        elif filter_type == 'highpass':
            cutoff = min(cutoff1, nyq * 0.9)
            normal = cutoff / nyq
            normal = max(0.05, min(0.95, normal))
            print(f"Highpass: {cutoff}Hz, norm={normal:.4f}")
            b, a = signal.butter(order, normal, btype='high')
            filtered = signal.filtfilt(b, a, sig)
            
        elif filter_type == 'bandpass':
            if cutoff1 >= cutoff2:
                return sig.tolist()
            low = min(cutoff1, nyq * 0.9)
            high = min(cutoff2, nyq * 0.9)
            normal = [low/nyq, high/nyq]
            normal = [max(0.05, min(0.95, n)) for n in normal]
            print(f"Bandpass: {low}-{high}Hz, norm={normal}")
            b, a = signal.butter(order, normal, btype='band')
            filtered = signal.filtfilt(b, a, sig)
            
        elif filter_type == 'bandstop':
            if cutoff1 >= cutoff2:
                return sig.tolist()
            low = min(cutoff1, nyq * 0.9)
            high = min(cutoff2, nyq * 0.9)
            normal = [low/nyq, high/nyq]
            normal = [max(0.05, min(0.95, n)) for n in normal]
            print(f"Bandstop: {low}-{high}Hz, norm={normal}")
            b, a = signal.butter(order, normal, btype='bandstop')
            filtered = signal.filtfilt(b, a, sig)
        else:
            return sig.tolist()
        
        # Check for NaN
        if np.any(np.isnan(filtered)):
            print("Filter produced NaN - returning original")
            return sig.tolist()
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(filtered))
        if max_val > 0.95:
            filtered = filtered * (0.9 / max_val)
            
        return filtered.tolist()
        
    except Exception as e:
        print(f"Filter error: {e}")
        return sig.tolist()
    
@app.route('/api/spectrum_analysis', methods=['POST'])
def spectrum_analysis():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400
            
        sig = np.array(data.get('signal', []))
        if len(sig) == 0:
            return jsonify({'error': 'No signal provided'}), 400
            
        fs = data.get('fs', 44100)
        
        # Limit to 512 samples for very fast computation
        sig = sig[:512]
        N = len(sig)
        
        if N < 10:
            return jsonify({'error': 'Signal too short'}), 400
        
        import time
        
        # Simple DFT with try-catch
        try:
            start_dft = time.time()
            dft_vals = []
            for k in range(min(N//2, 256)):  # Limit to 256 frequency bins
                sum_val = 0j
                for n in range(N):
                    sum_val += float(sig[n]) * np.exp(-2j * np.pi * k * n / N)
                dft_vals.append(sum_val)
            dft_time = time.time() - start_dft
            dft_mag = np.abs(dft_vals)
            all_freqs = np.fft.fftfreq(N, 1/fs)[:len(dft_vals)]
        except Exception as e:
            print(f"DFT error: {e}")
            dft_time = 0
            dft_mag = [0] * (N//2)
            all_freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
        
        # FFT (fast)
        try:
            start_fft = time.time()
            fft_vals = np.fft.fft(sig)
            fft_time = time.time() - start_fft
            fft_mag = np.abs(fft_vals)[:len(dft_mag)]
        except Exception as e:
            print(f"FFT error: {e}")
            fft_time = 0
            fft_mag = [0] * len(dft_mag)
        
        # Ensure same length
        min_len = min(len(all_freqs), len(dft_mag), len(fft_mag))
        all_freqs = all_freqs[:min_len]
        dft_mag = dft_mag[:min_len]
        fft_mag = fft_mag[:min_len]
        
        # Zoom to 0-2000 Hz
        max_freq = 2000
        zoom_indices = [i for i, f in enumerate(all_freqs) if f <= max_freq]
        
        if not zoom_indices:
            zoom_indices = list(range(min(50, len(all_freqs))))
        
        zoomed_freqs = [float(all_freqs[i]) for i in zoom_indices]
        zoomed_dft_mag = [float(dft_mag[i]) for i in zoom_indices]
        zoomed_fft_mag = [float(fft_mag[i]) for i in zoom_indices]
        
        # Calculate difference
        max_diff = 0
        if zoomed_dft_mag and zoomed_fft_mag:
            diff_array = [abs(zoomed_dft_mag[i] - zoomed_fft_mag[i]) for i in range(len(zoomed_dft_mag))]
            max_diff = max(diff_array)
        
        print(f"DFT: {dft_time:.4f}s, FFT: {fft_time:.4f}s, Speedup: {dft_time/fft_time:.1f}x" if fft_time > 0 else "FFT calculation done")
        
        return jsonify({
            'freqs': zoomed_freqs,
            'dft_mag': zoomed_dft_mag,
            'fft_mag': zoomed_fft_mag,
            'dft_time': dft_time,
            'fft_time': fft_time,
            'speedup': dft_time / fft_time if fft_time > 0 else 0,
            'max_difference': max_diff
        })
        
    except Exception as e:
        print(f"Spectrum analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def compute_fft(sig, fs):
    N = len(sig)
    if N == 0:
        return [], []
    fft_vals = np.fft.fft(sig)
    mag = np.abs(fft_vals)[:N//2]
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
    return freqs.tolist(), mag.tolist()

def apply_window(sig, window_type):
    N = len(sig)
    if window_type == 'rectangular':
        win = np.ones(N)
    elif window_type == 'hamming':
        win = np.hamming(N)
    elif window_type == 'hann':
        win = np.hanning(N)
    elif window_type == 'blackman':
        win = np.blackman(N)
    else:
        win = np.ones(N)
    return (np.array(sig) * win).tolist()

def ztransform_sequence(seq_str):
    try:
        coeffs = [float(x.strip()) for x in seq_str.split(',')]
        if not coeffs:
            return None
        z = symbols('z')
        H = sum(coeffs[i] * z**(-i) for i in range(len(coeffs)))
        H = together(H)
        num, den = sp.fraction(H)
        zeros = sp.solve(num, z)
        poles = sp.solve(den, z)
        zeros_c = [complex(zero) for zero in zeros]
        poles_c = [complex(pole) for pole in poles]
        return {
            'expression': sp.latex(H),
            'zeros': [(z.real, z.imag) for z in zeros_c],
            'poles': [(p.real, p.imag) for p in poles_c]
        }
    except:
        return None

# ---------- Routes ----------
@app.route('/')
def dashboard():
    projects = load_projects()
    return render_template('dashboard.html', projects=projects)

@app.route('/api/projects', methods=['GET'])
def get_projects():
    projects = load_projects()
    projects_list = [{
        'id': pid,
        'name': p['name'],
        'status': p.get('status', 'current'),
        'created_at': p['created_at']
    } for pid, p in projects.items()]
    return jsonify(projects_list)

@app.route('/create_project', methods=['POST'])
def create_project():
    data = request.get_json()
    name = data.get('name', 'Untitled Project')
    location = data.get('location', '')
    project_id = str(uuid.uuid4())[:8]
    projects = load_projects()
    projects[project_id] = {
        'id': project_id,
        'name': name,
        'status': 'current',
        'created_at': datetime.now().isoformat(),
        'location': location,
        'params': {
            'fs': 44100,
            'duration': 2.0,
            'freq': 440.0,
            'amp': 1.0,
            'phase': 0.0,
            'waveform': 'sine',
            'filter_type': 'lowpass',
            'cutoff1': 1000.0,
            'cutoff2': 4000.0,
            'order': 4,
            'window': 'rectangular',
            'volume': 1.0,
            'sample_fs': 1000.0,
            'alias_freq': 440.0,
            'z_sequence': '1, -0.5, 0.25'
        }
    }
    save_projects(projects)
    return jsonify({'id': project_id})

@app.route('/delete_project/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    projects = load_projects()
    if project_id in projects:
        del projects[project_id]
        save_projects(projects)
        return jsonify({'success': True})
    return jsonify({'success': False}), 404

@app.route('/api/export_project/<project_id>', methods=['GET'])
def export_project(project_id):
    projects = load_projects()
    if project_id not in projects:
        return jsonify({'error': 'Project not found'}), 404
    project_data = projects[project_id]
    response = Response(json.dumps(project_data, indent=2), mimetype='application/json')
    response.headers['Content-Disposition'] = f'attachment; filename=project_{project_id}.json'
    return response

@app.route('/project/<project_id>')
def console(project_id):
    projects = load_projects()
    if project_id not in projects:
        return redirect(url_for('dashboard'))
    project = projects[project_id]
    return render_template('console.html', project=project)

@app.route('/api/save_project/<project_id>', methods=['POST'])
def save_project(project_id):
    data = request.get_json()
    projects = load_projects()
    if project_id not in projects:
        return jsonify({'error': 'Project not found'}), 404
    projects[project_id]['params'].update(data)
    save_projects(projects)
    return jsonify({'success': True})

@app.route('/api/load_project/<project_id>', methods=['GET'])
def load_project(project_id):
    projects = load_projects()
    if project_id not in projects:
        return jsonify({'error': 'Project not found'}), 404
    return jsonify(projects[project_id]['params'])

# ---------- DSP Processing Endpoints ----------
@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.get_json()
    t, sig = generate_signal(
        data['fs'], data['duration'], data['freq'],
        data['amp'], data['phase'], data['waveform']
    )
    return jsonify({'t': t, 'signal': sig})

@app.route('/api/process', methods=['POST'])
def api_process():
    data = request.get_json()
    t, sig = generate_signal(
        data['fs'], data['duration'], data['freq'],
        data['amp'], data['phase'], data['waveform']
    )
    filtered = apply_filter(sig, data['fs'], data['filter_type'],
                            data['cutoff1'], data['cutoff2'], data['order'])
    windowed = apply_window(filtered, data['window'])
    processed = np.array(windowed) * data['volume']
    # Apply pan and EQ? (optional, but frontend may send)
    freqs, mag = compute_fft(processed, data['fs'])
    return jsonify({
        't': t,
        'original': sig,
        'filtered': filtered,
        'windowed': windowed,
        'processed': processed.tolist(),
        'freqs': freqs,
        'mag': mag
    })

@app.route('/api/process_with_signal', methods=['POST'])
def api_process_with_signal():
    data = request.get_json()
    sig = np.array(data.get('signal', []))
    if len(sig) == 0:
        return jsonify({'error': 'No signal provided'}), 400
    fs = data.get('fs', 44100)
    
    # Extract parameters
    filter_enable = data.get('filter_enable', False)
    filter_type = data.get('filter_type', 'lowpass')
    cutoff1 = data.get('cutoff1', 1000)
    cutoff2 = data.get('cutoff2', 4000)
    order = data.get('order', 4)
    window_enable = data.get('window_enable', False)
    window_type = data.get('window', 'rectangular')
    volume = data.get('volume', 1.0)
    pan = data.get('pan', 0.0)
    bass_gain = data.get('bass', 0.0)
    mid_gain = data.get('mid', 0.0)
    treble_gain = data.get('treble', 0.0)
    
    print(f"Processing: volume={volume}, pan={pan}, bass={bass_gain}, mid={mid_gain}, treble={treble_gain}")
    
    # Make a copy to avoid modifying original
    sig = sig.copy()
    
    # Apply filter if enabled (use lower order for stability)
    if filter_enable:
        filter_order = min(order, 2)  # Limit to order 2 for stability
        print(f"Using filter order: {filter_order} (requested: {order})")
        sig = np.array(apply_filter(sig.tolist(), fs, filter_type, cutoff1, cutoff2, filter_order))
    
    # Apply EQ safely - only if gain is non-zero
    try:
        if bass_gain != 0:
            sig = apply_low_shelf(sig, fs, bass_gain, cutoff=100)
            print(f"Bass applied: {bass_gain}dB")
    except Exception as e:
        print(f"Bass filter error: {e}")
    
    try:
        if mid_gain != 0:
            sig = apply_peak_eq(sig, fs, mid_gain, freq=1000, Q=1.0)
            print(f"Mid applied: {mid_gain}dB")
    except Exception as e:
        print(f"Mid filter error: {e}")
    
    try:
        if treble_gain != 0:
            sig = apply_high_shelf(sig, fs, treble_gain, cutoff=5000)
            print(f"Treble applied: {treble_gain}dB")
    except Exception as e:
        print(f"Treble filter error: {e}")
    
    # Apply window if enabled
    if window_enable:
        sig = np.array(apply_window(sig.tolist(), window_type))
    
    # Apply volume
    sig = sig * volume
    
    # Apply pan
    if len(sig.shape) == 1:
        if pan == 0:
            stereo_sig = np.column_stack((sig, sig))
        elif pan < 0:
            right_gain = 1 + pan
            stereo_sig = np.column_stack((sig, sig * right_gain))
        else:
            left_gain = 1 - pan
            stereo_sig = np.column_stack((sig * left_gain, sig))
        sig = stereo_sig
    
    # Prevent clipping
    max_val = np.max(np.abs(sig))
    if max_val > 0.95:
        sig = sig * (0.95 / max_val)
        print(f"Clipping prevented, reduced by factor {0.95/max_val:.3f}")
    
    # For FFT, use left channel only
    if len(sig.shape) == 2:
        sig_mono = sig[:, 0]
    else:
        sig_mono = sig
    
    freqs, mag = compute_fft(sig_mono.tolist(), fs)
    
    return jsonify({
        'processed': sig.tolist(),
        'freqs': freqs,
        'mag': mag
    })

@app.route('/api/aliasing', methods=['POST'])
def api_aliasing():
    data = request.get_json()
    fs_signal = data.get('signal_freq', 440)
    fs_sample = data.get('sample_freq', 1000)
    fs_cont = 100000
    t_cont = np.linspace(0, 0.02, int(fs_cont*0.02), endpoint=False)
    sig_cont = np.sin(2*np.pi*fs_signal*t_cont)
    interval = max(1, int(fs_cont / fs_sample))
    t_samp = t_cont[::interval]
    sig_samp = sig_cont[::interval]
    return jsonify({
        't_cont': t_cont.tolist(),
        'sig_cont': sig_cont.tolist(),
        't_samp': t_samp.tolist(),
        'sig_samp': sig_samp.tolist(),
        'aliasing': fs_sample < 2*fs_signal
    })

@app.route('/api/ztransform', methods=['POST'])
def api_ztransform():
    data = request.get_json()
    seq = data.get('sequence', '1, -0.5, 0.25')
    result = ztransform_sequence(seq)
    if result:
        # Calculate frequency response from Z-transform
        coeffs = [float(x.strip()) for x in seq.split(',')]
        if coeffs:
            # Evaluate on unit circle (z = e^(jω))
            w = np.linspace(0, np.pi, 500)  # 0 to π radians (0 to Nyquist)
            z = np.exp(1j * w)
            # H(z) = sum(coeffs[n] * z^(-n))
            H = np.zeros(len(w), dtype=complex)
            for n, c in enumerate(coeffs):
                H += c * z**(-n)
            magnitude = np.abs(H)
            phase = np.angle(H)
            freqs = w * (44100 / (2 * np.pi))  # Convert to Hz
            
            # LIMIT to 5000 Hz for better display (not full 22kHz)
            max_freq = 5000
            limit_indices = [i for i, f in enumerate(freqs) if f <= max_freq]
            
            result['freq_response'] = {
                'freqs': freqs[limit_indices].tolist(),
                'magnitude': magnitude[limit_indices].tolist(),
                'phase': phase[limit_indices].tolist()
            }
            result['coefficients'] = coeffs
            result['explanation'] = {
                'stability': 'Stable' if all(abs(p[0] + 1j*p[1]) < 1 for p in result['poles']) else 'Unstable',
                'num_poles': len(result['poles']),
                'num_zeros': len(result['zeros'])
            }
        return jsonify(result)
    else:
        return jsonify({'error': 'Invalid sequence'}), 400

@app.route('/api/dft_fft_compare', methods=['POST'])
def api_dft_fft_compare():
    data = request.get_json()
    sig = data.get('signal', [])
    if len(sig) > 4096:
        sig = sig[:4096]
    N = len(sig)
    if N == 0:
        return jsonify({'error': 'Empty signal'})
    import time
    start = time.time()
    dft = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            dft[k] += sig[n] * np.exp(-2j*np.pi*k*n/N)
    dft_time = time.time() - start
    start = time.time()
    fft = np.fft.fft(sig)
    fft_time = time.time() - start
    return jsonify({
        'dft_time': dft_time,
        'fft_time': fft_time,
        'speedup': dft_time/fft_time if fft_time > 0 else 0,
        'max_diff': float(np.max(np.abs(dft - fft)))
    })

import subprocess

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.lower().endswith('.wav'):
        return jsonify({'error': 'Only WAV files are supported'}), 400
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Read using scipy
        fs, data = wavfile.read(tmp_path)
        
        # Convert to float32 and normalize
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32767.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128) / 128.0
        else:
            # If float, assume already in [-1,1]
            data = data.astype(np.float32)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        t = np.linspace(0, len(data)/fs, len(data), endpoint=False).tolist()
        
        # Clean up
        os.unlink(tmp_path)
        
        return jsonify({
            't': t,
            'signal': data.tolist(),
            'fs': fs,
            'wav_url': '',
            'duration': len(data) / fs
        })
        
    except Exception as e:
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass
        # Return detailed error message to client
        import traceback
        error_details = traceback.format_exc()
        print(error_details)  # This will appear in Render logs if they ever show up
        return jsonify({'error': f'Failed to read WAV: {str(e)}', 'traceback': error_details}), 500

@app.route('/api/audio/<file_id>.wav')
def serve_audio(file_id):
    if hasattr(app, 'temp_files') and file_id in app.temp_files:
        return send_file(app.temp_files[file_id], mimetype='audio/wav')
    return 'File not found', 404

@app.route('/api/process_and_play', methods=['POST'])
def process_and_play():
    data = request.get_json()
    
    print("=" * 50)
    print("PROCESS_AND_PLAY CALLED")
    print(f"Received keys: {list(data.keys())}")
    
    # Check if we're receiving an already-processed signal from frontend
    sig = np.array(data.get('processed_signal', []))
    
    # If no processed_signal, then use the original signal
    if len(sig) == 0:
        sig = np.array(data.get('signal', []))
        print("Using original signal")
    else:
        print("Using pre-processed signal from frontend")
    
    if len(sig) == 0:
        return jsonify({'error': 'No signal'}), 400
    
    fs = data.get('fs', 44100)
    
    # Extract parameters
    filter_enable = data.get('filter_enable', False)
    filter_type = data.get('filter_type', 'lowpass')
    cutoff1 = data.get('cutoff1', 1000)
    cutoff2 = data.get('cutoff2', 4000)
    order = data.get('order', 4)
    window_enable = data.get('window_enable', False)
    window_type = data.get('window', 'rectangular')
    volume = data.get('volume', 1.0)
    pan_value = data.get('pan', 0.0)
    bass_gain = data.get('bass', 0.0)
    mid_gain = data.get('mid', 0.0)
    treble_gain = data.get('treble', 0.0)
    
    print(f"Parameters:")
    print(f"   - Volume: {volume}")
    print(f"   - Pan: {pan_value}")
    print(f"   - Bass: {bass_gain} dB")
    print(f"   - Mid: {mid_gain} dB")
    print(f"   - Treble: {treble_gain} dB")
    print(f"   - Filter: {filter_enable}")
    print(f"   - Window: {window_enable}")
    
    # Ensure signal is float and in range [-1, 1]
    if sig.dtype == np.int16:
        sig = sig.astype(np.float32) / 32767.0
        print("Converted int16 to float32")
    
    print(f"Initial signal shape: {sig.shape}, dtype: {sig.dtype}")
    print(f"Initial signal range: [{np.min(sig):.3f}, {np.max(sig):.3f}]")
    
    # Apply filter if enabled (use lower order for stability)
    if filter_enable:
        filter_order = min(order, 2)  # Limit to order 2 for stability
        print(f"Using filter order: {filter_order} (requested: {order})")
        sig = np.array(apply_filter(sig.tolist(), fs, filter_type, cutoff1, cutoff2, filter_order))
        print(f"Filter applied - new range: [{np.min(sig):.3f}, {np.max(sig):.3f}]")
    
    # Apply EQ in correct order (Bass -> Mid -> Treble)
    if bass_gain != 0:
        original_range = [np.min(sig), np.max(sig)]
        sig = apply_low_shelf(sig, fs, bass_gain, cutoff=100)
        print(f"Bass EQ applied (gain: {bass_gain}dB) - range changed from [{original_range[0]:.3f}, {original_range[1]:.3f}] to [{np.min(sig):.3f}, {np.max(sig):.3f}]")
    
    if mid_gain != 0:
        sig = apply_peak_eq(sig, fs, mid_gain, freq=1000, Q=1.0)
        print(f"Mid EQ applied (gain: {mid_gain}dB)")
    
    if treble_gain != 0:
        sig = apply_high_shelf(sig, fs, treble_gain, cutoff=5000)
        print(f"Treble EQ applied (gain: {treble_gain}dB)")
    
    # Apply window if enabled
    if window_enable:
        sig = np.array(apply_window(sig.tolist(), window_type))
        print(f"Window applied")
    
    # Apply volume
    sig = sig * volume
    print(f"Volume applied (factor: {volume}) - new range: [{np.min(sig):.3f}, {np.max(sig):.3f}]")
    
    # Apply pan (convert to stereo)
    sig = apply_pan(sig, pan_value)
    print(f"Pan applied (value: {pan_value}), output shape: {sig.shape}")
    
    # Normalize only if clipping would occur
    max_val = np.max(np.abs(sig))
    if max_val > 1.0:
        sig = sig * (0.95 / max_val)
        print(f"Clipping detected! Reduced by factor {0.95/max_val:.3f}")
    else:
        print(f"No clipping (peak: {max_val:.3f})")
    
    # Final check of audio levels
    print(f"Final signal stats:")
    print(f"   - Min: {np.min(sig):.4f}")
    print(f"   - Max: {np.max(sig):.4f}")
    print(f"   - Mean: {np.mean(sig):.4f}")
    print(f"   - Std: {np.std(sig):.4f}")
    
    # Convert to int16 for WAV
    data_int16 = (sig * 32767).astype(np.int16)
    
    # Save to temp file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wavfile.write(temp_wav.name, fs, data_int16)
    print(f"Saved WAV to: {temp_wav.name}")
    print("=" * 50)
    
    return send_file(temp_wav.name, mimetype='audio/wav')

def apply_pan(sig, pan_value):
    """
    Convert mono to stereo with pan control using constant power panning.
    pan_value: -1 = full left, 0 = center, +1 = full right
    """
    # If input is already stereo, reduce to mono first
    if len(sig.shape) == 2 and sig.shape[1] == 2:
        sig = np.mean(sig, axis=1)
    
    # Ensure sig is 1D
    sig = sig.flatten()
    
    if pan_value == 0:
        # Center: both channels equal
        return np.column_stack((sig, sig))
    
    # Use constant power panning for smoother transition
    # Convert pan from [-1, 1] to angle [0, π/2]
    angle = (pan_value + 1) * (np.pi / 4)  # Maps -1->0, 0->π/4, 1->π/2
    
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)
    
    # For negative pan (left), swap gains
    if pan_value < 0:
        left_gain, right_gain = right_gain, left_gain
    
    left = sig * left_gain
    right = sig * right_gain
    
    print(f"Pan gains - Left: {left_gain:.3f}, Right: {right_gain:.3f}")
    
    return np.column_stack((left, right))

def apply_image_filter(image, filter_type, intensity=2.0):
    """
    Applies a specified filter to an image.
    """
    # Convert PIL Image to OpenCV format (BGR)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    if filter_type == 'gaussian':
        # Gaussian Blur: smooths image using a Gaussian kernel[reference:4]
        ksize = max(1, int(intensity) if intensity % 2 == 1 else int(intensity) + 1)
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif filter_type == 'average':
        # Average Blur: replaces pixel with average of its neighbors[reference:5]
        ksize = max(1, int(intensity) if intensity % 2 == 1 else int(intensity) + 1)
        img = cv2.blur(img, (ksize, ksize))
    elif filter_type == 'median':
        # Median Blur: replaces pixel with median of its neighbors, good for noise[reference:6]
        ksize = max(1, int(intensity) if intensity % 2 == 1 else int(intensity) + 1)
        img = cv2.medianBlur(img, ksize)
    elif filter_type == 'sobel':
        # Sobel Edge Detection: detects edges using gradient approximation[reference:7]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.hypot(sobelx, sobely)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        img = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    elif filter_type == 'prewitt':
        # Prewitt Edge Detection: similar to Sobel with a different kernel[reference:8]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewittx = cv2.filter2D(gray, -1, kernelx)
        prewitty = cv2.filter2D(gray, -1, kernely)
        prewitt = np.hypot(prewittx, prewitty)
        prewitt = np.uint8(np.clip(prewitt, 0, 255))
        img = cv2.cvtColor(prewitt, cv2.COLOR_GRAY2BGR)
    elif filter_type == 'laplacian':
        # Laplacian Edge Detection: finds edges by looking for zero crossings[reference:9][reference:10]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.clip(np.absolute(laplacian), 0, 255))
        img = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    elif filter_type == 'sharpen':
        # Image Sharpening: enhances edges using a sharpening kernel[reference:11][reference:12]
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
    return img

if __name__ == '__main__':
    app.run(debug=True)