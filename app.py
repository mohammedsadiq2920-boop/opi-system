"""
OPI - Oil Production Improvement System
نظام تحسين الإنتاج النفطي باستخدام الشبكات العصبية
"""

import os
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template, g

# --- Neural Network (pure numpy, no tensorflow needed) ---
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from scipy.optimize import minimize

app = Flask(__name__)
app.config['DATABASE'] = 'opi_data.db'
app.config['MODEL_PATH'] = 'model.pkl'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ─────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    db = sqlite3.connect(app.config['DATABASE'])
    db.execute("""
        CREATE TABLE IF NOT EXISTS production_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            total_liquid REAL,
            oil_production REAL,
            water_production REAL,
            water_cut REAL,
            upstream_pres REAL,
            downstream_pres REAL,
            choke_size REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    db.close()


# ─────────────────────────────────────────────
#  MODEL MANAGER
# ─────────────────────────────────────────────

class OilProductionModel:
    """
    الشبكة العصبية لتحسين الإنتاج النفطي
    
    المدخلات (7): سائل، نفط، ماء، نسبة ماء، ضغط أعلى، ضغط أدنى، خانق
    المخرجات (5): نفط_متوقع، ماء_متوقع، نسبة_ماء_متوقعة، ضغط_أعلى_مقترح، ضغط_أدنى_مقترح
    """

    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False
        self.train_samples = 0
        self.score = 0.0
        self.last_trained = None
        self.epochs = 500
        self.batch_size = 32
        # حدود المتغيرات للتحسين
        self.bounds = {
            'upstream_pres':   (0, 58),
            'downstream_pres': (0, 38.6),
            'choke_size':      (0, 128)
        }

    def _build_model(self):
        return MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=self.epochs,
            batch_size=self.batch_size,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            learning_rate_init=0.001,
            warm_start=False
        )

    def train(self, df: pd.DataFrame):
        """تدريب النموذج على البيانات"""
        required_cols = ['total_liquid', 'oil_production', 'water_production',
                         'water_cut', 'upstream_pres', 'downstream_pres', 'choke_size']
        df = df.dropna(subset=required_cols)
        if len(df) < 10:
            raise ValueError("البيانات غير كافية للتدريب (أقل من 10 سجلات)")

        X = df[required_cols].values.astype(float)
        # المخرجات: نفط، ماء، نسبة ماء، ضغط أعلى، ضغط أدنى
        y = df[['oil_production', 'water_production', 'water_cut',
                 'upstream_pres', 'downstream_pres']].values.astype(float)

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        self.model = self._build_model()
        self.model.fit(X_scaled, y_scaled)

        y_pred = self.scaler_y.inverse_transform(self.model.predict(X_scaled))
        # R² score يدوياً
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        self.score = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        self.is_trained = True
        self.train_samples = len(df)
        self.last_trained = datetime.now().strftime('%Y-%m-%d %H:%M')

    def predict(self, features: dict) -> dict:
        """التنبؤ بالمخرجات للمدخلات المعطاة"""
        if not self.is_trained:
            raise RuntimeError("النموذج غير مدرب")

        cols = ['total_liquid', 'oil_production', 'water_production',
                'water_cut', 'upstream_pres', 'downstream_pres', 'choke_size']
        X = np.array([[features.get(c, 0) or 0 for c in cols]], dtype=float)
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.model.predict(X_scaled)
        y = self.scaler_y.inverse_transform(y_scaled)[0]

        return {
            'oil_production':   float(y[0]),
            'water_production': float(y[1]),
            'water_cut':        float(np.clip(y[2], 0, 100)),
            'upstream_pres':    float(np.clip(y[3], *self.bounds['upstream_pres'])),
            'downstream_pres':  float(np.clip(y[4], *self.bounds['downstream_pres']))
        }

    def optimize(self, features: dict) -> dict:
        """
        البحث عن أفضل قيم للضغطين وفتحة الخانق لتعظيم النفط وتقليل الماء
        """
        if not self.is_trained:
            raise RuntimeError("النموذج غير مدرب")

        cols = ['total_liquid', 'oil_production', 'water_production',
                'water_cut', 'upstream_pres', 'downstream_pres', 'choke_size']

        def objective(params):
            upstream, downstream, choke = params
            f = dict(features)
            f['upstream_pres']   = upstream
            f['downstream_pres'] = downstream
            f['choke_size']      = choke
            X = np.array([[f.get(c, 0) or 0 for c in cols]], dtype=float)
            X_sc = self.scaler_X.transform(X)
            y_sc = self.model.predict(X_sc)
            y = self.scaler_y.inverse_transform(y_sc)[0]
            oil = y[0]
            wc  = np.clip(y[2], 0, 100)
            # نريد تعظيم النفط وتقليل نسبة الماء → minimize (−oil + wc_penalty)
            return -oil + 0.5 * wc

        # نقطة البداية: القيم الحالية
        x0 = [
            features.get('upstream_pres',   30) or 30,
            features.get('downstream_pres', 20) or 20,
            features.get('choke_size',       32) or 32
        ]
        bounds = [
            self.bounds['upstream_pres'],
            self.bounds['downstream_pres'],
            self.bounds['choke_size']
        ]

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-9})

        opt_up, opt_down, opt_choke = result.x

        # تنبؤ بالمخرجات عند القيم المثلى
        f_opt = dict(features)
        f_opt['upstream_pres']   = opt_up
        f_opt['downstream_pres'] = opt_down
        f_opt['choke_size']      = opt_choke
        pred = self.predict(f_opt)

        return {
            'upstream_pres':   float(np.clip(opt_up,    *self.bounds['upstream_pres'])),
            'downstream_pres': float(np.clip(opt_down,  *self.bounds['downstream_pres'])),
            'choke_size':      float(np.clip(opt_choke, *self.bounds['choke_size'])),
            'predicted_oil':        pred['oil_production'],
            'predicted_water':      pred['water_production'],
            'predicted_water_cut':  pred['water_cut'],
        }


# ─────────────────────────────────────────────
#  GLOBAL MODEL INSTANCE
# ─────────────────────────────────────────────

model_instance = OilProductionModel()

def save_model():
    with open(app.config['MODEL_PATH'], 'wb') as f:
        pickle.dump(model_instance, f)

def load_model():
    global model_instance
    if os.path.exists(app.config['MODEL_PATH']):
        with open(app.config['MODEL_PATH'], 'rb') as f:
            model_instance = pickle.load(f)


# ─────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────

COL_MAP = {
    'تاريخ':              'date',
    'السائل المستخرج':   'total_liquid',
    'النفط المستخرج':    'oil_production',
    'الماء المستخرج':    'water_production',
    'نسبة الماء ':       'water_cut',
    'نسبة الماء':        'water_cut',
    'الضغط الأعلى':      'upstream_pres',
    'الضغط الأدنى ':     'downstream_pres',
    'الضغط الأدنى':      'downstream_pres',
    'فتحة الخانق':       'choke_size',
    'THEDATE':            'date',
    'DAILY LIQUID':       'total_liquid',
    'DAILYOIL':           'oil_production',
    'QW':                 'water_production',
    'WATER CUT':          'water_cut',
    'UPSTREAMPRES':       'upstream_pres',
    'DOWNSTREAMPRES':     'downstream_pres',
    'CHOKESIZE':          'choke_size',
}

def parse_excel(filepath) -> pd.DataFrame:
    df = pd.read_excel(filepath, header=0)
    # إذا كان الصف الأول يحتوي على أسماء إنجليزية بديلة
    if df.iloc[0].astype(str).str.contains('DAILY|OIL|WATER|PRES|CHOKE|DATE', case=False).any():
        df = df.iloc[1:].reset_index(drop=True)

    df.rename(columns={c: COL_MAP.get(c.strip(), c) for c in df.columns}, inplace=True)
    df.rename(columns=lambda c: COL_MAP.get(c.strip(), c), inplace=True)

    num_cols = ['total_liquid', 'oil_production', 'water_production',
                'water_cut', 'upstream_pres', 'downstream_pres', 'choke_size']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

    return df

def db_to_dataframe() -> pd.DataFrame:
    db = sqlite3.connect(app.config['DATABASE'])
    df = pd.read_sql("SELECT * FROM production_data", db)
    db.close()
    return df


# ─────────────────────────────────────────────
#  PAGE ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize')
def optimize_page():
    return render_template('optimize.html')

@app.route('/data')
def data_page():
    return render_template('data.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/model')
def model_page():
    return render_template('model.html')


# ─────────────────────────────────────────────
#  API ROUTES
# ─────────────────────────────────────────────

@app.route('/api/stats')
def api_stats():
    db = get_db()
    row = db.execute("""
        SELECT COUNT(*) as cnt,
               AVG(water_cut) as avg_wc,
               AVG(oil_production) as avg_oil
        FROM production_data
    """).fetchone()
    return jsonify({
        'total_records': row['cnt'],
        'avg_water_cut': row['avg_wc'] or 0,
        'avg_oil':       row['avg_oil'] or 0,
        'model_ready':   model_instance.is_trained
    })


@app.route('/api/data')
def api_data():
    db = get_db()
    rows = db.execute("SELECT * FROM production_data ORDER BY date DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/data/<int:row_id>', methods=['DELETE'])
def api_delete_row(row_id):
    db = get_db()
    db.execute("DELETE FROM production_data WHERE id=?", (row_id,))
    db.commit()
    return jsonify({'success': True})


@app.route('/api/data/all', methods=['DELETE'])
def api_delete_all():
    db = get_db()
    db.execute("DELETE FROM production_data")
    db.commit()
    return jsonify({'success': True})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'لم يتم إرفاق ملف'}), 400

    file = request.files['file']
    if not file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({'error': 'يجب أن يكون الملف بصيغة Excel'}), 400

    tmp = f'/tmp/upload_{datetime.now().timestamp()}.xlsx'
    file.save(tmp)

    try:
        df = parse_excel(tmp)
    except Exception as e:
        return jsonify({'error': f'خطأ في قراءة الملف: {str(e)}'}), 400
    finally:
        os.remove(tmp)

    required = ['oil_production', 'water_cut']
    for c in required:
        if c not in df.columns:
            return jsonify({'error': f'العمود المطلوب غير موجود: {c}'}), 400

    db = get_db()
    added = 0
    for _, row in df.iterrows():
        vals = {c: (None if pd.isna(row.get(c, np.nan)) else row.get(c)) for c in
                ['date', 'total_liquid', 'oil_production', 'water_production',
                 'water_cut', 'upstream_pres', 'downstream_pres', 'choke_size']}
        if vals['oil_production'] is None:
            continue
        db.execute("""
            INSERT INTO production_data
            (date, total_liquid, oil_production, water_production,
             water_cut, upstream_pres, downstream_pres, choke_size)
            VALUES (:date, :total_liquid, :oil_production, :water_production,
                    :water_cut, :upstream_pres, :downstream_pres, :choke_size)
        """, vals)
        added += 1
    db.commit()

    # إعادة التدريب التلقائي
    retrain_msg = ''
    try:
        full_df = db_to_dataframe()
        model_instance.train(full_df)
        save_model()
        retrain_msg = f'تم إعادة تدريب النموذج بنجاح على {len(full_df)} سجل.'
    except Exception as e:
        retrain_msg = f'تنبيه: {str(e)}'

    return jsonify({'added_count': added, 'retrain_message': retrain_msg})


@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'البيانات مفقودة'}), 400

    required = ['oil_production', 'water_cut']
    for f in required:
        if data.get(f) is None:
            return jsonify({'error': f'الحقل مطلوب: {f}'}), 400

    # تعبئة القيم الناقصة من متوسط البيانات
    note = None
    if not model_instance.is_trained:
        # محاولة التدريب من قاعدة البيانات
        try:
            df = db_to_dataframe()
            model_instance.train(df)
            save_model()
        except Exception as e:
            return jsonify({'error': f'النموذج غير جاهز: {str(e)}'}), 503

    # تعبئة الحقول الناقصة بالمتوسطات
    db = get_db()
    avgs = db.execute("""
        SELECT AVG(total_liquid) al, AVG(upstream_pres) up,
               AVG(downstream_pres) dp, AVG(choke_size) cs
        FROM production_data
    """).fetchone()

    features = {
        'total_liquid':    data.get('total_liquid')    or (avgs['al'] or 1200),
        'oil_production':  data['oil_production'],
        'water_production':data.get('water_production') or 0,
        'water_cut':       data['water_cut'],
        'upstream_pres':   data.get('upstream_pres')   or (avgs['up'] or 30),
        'downstream_pres': data.get('downstream_pres') or (avgs['dp'] or 21),
        'choke_size':      data.get('choke_size')      or (avgs['cs'] or 32),
    }

    if any(data.get(k) is None for k in ['upstream_pres', 'downstream_pres', 'choke_size']):
        note = 'بعض معاملات التشغيل غير مُدخلة، تم استخدام المتوسطات التاريخية.'

    try:
        optimized = model_instance.optimize(features)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # حساب التحسين
    oil_gain = optimized['predicted_oil'] - features['oil_production']
    wc_reduction = features['water_cut'] - optimized['predicted_water_cut']

    # مستوى الثقة مبني على درجة النموذج وحجم البيانات
    confidence = min(0.95, max(0.3, model_instance.score * 0.9 +
                               min(model_instance.train_samples / 1000, 0.1)))

    return jsonify({
        'optimized': {
            'upstream_pres':       round(optimized['upstream_pres'],   2),
            'downstream_pres':     round(optimized['downstream_pres'], 2),
            'choke_size':          round(optimized['choke_size'],       1),
            'predicted_oil':       round(optimized['predicted_oil'],    2),
            'predicted_water_cut': round(optimized['predicted_water_cut'], 2),
        },
        'improvement': {
            'oil_gain':          round(max(oil_gain, 0),       2),
            'water_cut_reduction': round(max(wc_reduction, 0), 2),
        },
        'confidence': round(confidence, 3),
        'note': note
    })


@app.route('/api/model/info')
def api_model_info():
    return jsonify({
        'ready':        model_instance.is_trained,
        'train_samples':model_instance.train_samples,
        'score':        round(model_instance.score, 4) if model_instance.is_trained else None,
        'last_trained': model_instance.last_trained,
        'epochs':       model_instance.epochs,
        'batch_size':   model_instance.batch_size,
    })


@app.route('/api/model/retrain', methods=['POST'])
def api_retrain():
    try:
        df = db_to_dataframe()
        if len(df) < 10:
            return jsonify({'error': 'البيانات غير كافية (أقل من 10 سجلات)'}), 400
        model_instance.train(df)
        save_model()
        return jsonify({
            'success': True,
            'samples': model_instance.train_samples,
            'score':   round(model_instance.score, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────

def startup():
    init_db()
    load_model()

    # تحميل ملف Excel المرفق إذا كانت قاعدة البيانات فارغة
    db = sqlite3.connect(app.config['DATABASE'])
    count = db.execute("SELECT COUNT(*) FROM production_data").fetchone()[0]
    db.close()

    if count == 0:
        excel_file = 'تحسين_الانتاج.xlsx'
        if os.path.exists(excel_file):
            print("📥 تحميل البيانات من ملف Excel...")
            try:
                df = parse_excel(excel_file)
                conn = sqlite3.connect(app.config['DATABASE'])
                added = 0
                for _, row in df.iterrows():
                    vals = {c: (None if pd.isna(row.get(c, np.nan)) else row.get(c)) for c in
                            ['date', 'total_liquid', 'oil_production', 'water_production',
                             'water_cut', 'upstream_pres', 'downstream_pres', 'choke_size']}
                    if vals.get('oil_production') is None:
                        continue
                    conn.execute("""
                        INSERT INTO production_data
                        (date, total_liquid, oil_production, water_production,
                         water_cut, upstream_pres, downstream_pres, choke_size)
                        VALUES (:date,:total_liquid,:oil_production,:water_production,
                                :water_cut,:upstream_pres,:downstream_pres,:choke_size)
                    """, vals)
                    added += 1
                conn.commit()
                conn.close()
                print(f"✅ تم تحميل {added} سجل")
            except Exception as e:
                print(f"⚠️  خطأ في تحميل Excel: {e}")

    # تدريب النموذج إذا لم يكن مدرباً
    if not model_instance.is_trained:
        db = sqlite3.connect(app.config['DATABASE'])
        df = pd.read_sql("SELECT * FROM production_data", db)
        db.close()
        if len(df) >= 10:
            print("🧠 تدريب الشبكة العصبية...")
            try:
                model_instance.train(df)
                save_model()
                print(f"✅ تم التدريب — دقة النموذج R²={model_instance.score:.3f}")
            except Exception as e:
                print(f"⚠️  خطأ في التدريب: {e}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    startup()
    print("\n🚀 النظام يعمل على: http://127.0.0.1:5000\n")
    port = int(os.environ.get('PORT', 5000))
app.run(debug=False, host='0.0.0.0', port=port)