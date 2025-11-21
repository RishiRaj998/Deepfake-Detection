import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

WORK_DIR = os.path.join(os.getcwd(), "processed_data")
REAL_PATH = os.path.join(os.getcwd(), "Video_detection", "DFD_original sequences")
FAKE_PATH = os.path.join(os.getcwd(), "Video_detection", "DFD_manipulated_sequences")
OUTPUT_FRAME_SIZE = (128, 128)
FRAME_COUNT = 10
MAX_VIDEOS = 200  # limit max videos for quick testing, adjust as needed
os.makedirs(WORK_DIR, exist_ok=True)

def path_check():
    if not os.path.exists(REAL_PATH) or not os.path.exists(FAKE_PATH):
        raise FileNotFoundError(
            f"Missing dataset folders.\nREAL_PATH: {REAL_PATH}\nFAKE_PATH: {FAKE_PATH}"
        )
path_check()

detector = MTCNN()
def extract_face(frame):
    results = detector.detect_faces(frame)
    if results:
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = frame[y:y+h, x:x+w]
        return cv2.resize(face, OUTPUT_FRAME_SIZE)
    return cv2.resize(frame, OUTPUT_FRAME_SIZE)

def extract_frames(video_path, output_size=(128,128), frame_count=10):
    cap = cv2.VideoCapture(video_path)
    frames, total_frames = [], int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // frame_count, 1)
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = extract_face(frame)
        frames.append(frame)
    cap.release()
    if len(frames) == frame_count:
        return np.array(frames)
    else:
        return np.zeros((frame_count, output_size[0], output_size[1], 3), dtype=np.uint8)

def save_arrays():
    print("Extracting and saving video arrays...")
    real_videos = os.listdir(REAL_PATH)[:MAX_VIDEOS]
    fake_videos = os.listdir(FAKE_PATH)[:MAX_VIDEOS]
    data, labels = [], []
    for files, lab, folder in [
        (real_videos, 0, REAL_PATH),
        (fake_videos, 1, FAKE_PATH)
    ]:
        for video_file in tqdm(files):
            path = os.path.join(folder, video_file)
            frames = extract_frames(path, output_size=OUTPUT_FRAME_SIZE, frame_count=FRAME_COUNT)
            if len(frames) == FRAME_COUNT:
                data.append(frames)
                labels.append(lab)
    data, labels = np.array(data), np.array(labels)
    np.save(os.path.join(WORK_DIR, "data.npy"), data)
    np.save(os.path.join(WORK_DIR, "labels.npy"), labels)

def load_or_create_data():
    if not (os.path.exists(os.path.join(WORK_DIR, "data.npy")) and os.path.exists(os.path.join(WORK_DIR, "labels.npy"))):
        save_arrays()
    data = np.load(os.path.join(WORK_DIR, "data.npy"))
    labels = np.load(os.path.join(WORK_DIR, "labels.npy"))
    print(f"Loaded dataset: {len(labels)} videos, Distribution REAL:{np.sum(labels==0)} FAKE:{np.sum(labels==1)}")
    return data, labels

data, labels = load_or_create_data()

def train_test_val_split(data, labels):
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0
    y_train, y_val, y_test = to_categorical(y_train, 2), to_categorical(y_val, 2), to_categorical(y_test, 2)

    print("Label distribution after split:")
    print(f"Train: {np.sum(np.argmax(y_train, axis=1) == 0)} REAL, {np.sum(np.argmax(y_train, axis=1) == 1)} FAKE")
    print(f"Val: {np.sum(np.argmax(y_val, axis=1) == 0)} REAL, {np.sum(np.argmax(y_val, axis=1) == 1)} FAKE")
    print(f"Test: {np.sum(np.argmax(y_test, axis=1) == 0)} REAL, {np.sum(np.argmax(y_test, axis=1) == 1)} FAKE")

    np.save(os.path.join(WORK_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(WORK_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(WORK_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(WORK_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(WORK_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(WORK_DIR, "y_test.npy"), y_test)

if not (os.path.exists(os.path.join(WORK_DIR, "X_train.npy")) and
        os.path.exists(os.path.join(WORK_DIR, "X_val.npy")) and
        os.path.exists(os.path.join(WORK_DIR, "X_test.npy"))):
    train_test_val_split(data, labels)

X_train = np.load(os.path.join(WORK_DIR, "X_train.npy"))
y_train = np.load(os.path.join(WORK_DIR, "y_train.npy"))
X_val = np.load(os.path.join(WORK_DIR, "X_val.npy"))
y_val = np.load(os.path.join(WORK_DIR, "y_val.npy"))
X_test = np.load(os.path.join(WORK_DIR, "X_test.npy"))
y_test = np.load(os.path.join(WORK_DIR, "y_test.npy"))

aug_data_path = os.path.join(WORK_DIR, "X_train_augmented.npy")
aug_labels_path = os.path.join(WORK_DIR, "y_train_augmented.npy")

if not (os.path.exists(aug_data_path) and os.path.exists(aug_labels_path)):
    print("Performing and saving augmented training data...")
    datagen = ImageDataGenerator(
        horizontal_flip=True, rotation_range=40, zoom_range=0.4, brightness_range=[0.5, 1.5],
        shear_range=0.4, width_shift_range=0.3, height_shift_range=0.3, channel_shift_range=10.0,
        fill_mode='nearest'
    )
    def augment_frames(frames):
        return np.array([datagen.random_transform(frame) for frame in frames])
    augmented_data, augmented_labels = [], []
    for i in tqdm(range(len(X_train))):
        augmented_data.append(augment_frames(X_train[i]))
        augmented_labels.append(y_train[i])
    X_train_augmented = np.concatenate((X_train, np.array(augmented_data)))
    y_train_augmented = np.concatenate((y_train, np.array(augmented_labels)))
    np.save(aug_data_path, X_train_augmented)
    np.save(aug_labels_path, y_train_augmented)

X_train_augmented = np.load(aug_data_path)
y_train_augmented = np.load(aug_labels_path)

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def compute_output_shape(self, input_shape):
        batch_size, h, w, c = input_shape
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        num_patches = num_patches_h * num_patches_w
        patch_dims = self.patch_size * self.patch_size * c
        return (batch_size, num_patches, patch_dims)

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def build(self, input_shape):
        self.projection.build(input_shape)
        self.position_embedding.build((input_shape[0], self.num_patches))
        super().build(input_shape)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def compute_output_shape(self, input_shape):
        batch_size, num_patches, _ = input_shape
        projection_dim = self.projection.units
        return (batch_size, num_patches, projection_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection.units
        })
        return config

def build_best_model(
    input_shape=(FRAME_COUNT, OUTPUT_FRAME_SIZE[0], OUTPUT_FRAME_SIZE[1], 3),
    patch_size=16,
    projection_dim=64,
    num_classes=2,
):
    inputs = layers.Input(shape=input_shape)
    num_patches = (OUTPUT_FRAME_SIZE[0] // patch_size) * (OUTPUT_FRAME_SIZE[1] // patch_size)

    patches = TimeDistributed(Patches(patch_size))(inputs)
    encoded_patches = TimeDistributed(PatchEncoder(num_patches, projection_dim))(patches)
    x = TimeDistributed(layers.LayerNormalization(epsilon=1e-6))(encoded_patches)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model_path = os.path.join(WORK_DIR, "deepfake_detection_vit_model_final.keras")
checkpoint_path = os.path.join(WORK_DIR, "deepfake_detection_vit_model.best.h5")
history_path = os.path.join(WORK_DIR, "training_history_vit.npy")

if not os.path.exists(model_path):
    model = build_best_model()
    model.summary()
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train_augmented, y_train_augmented,
        validation_data=(X_val, y_val),
        epochs=50, batch_size=8,
        class_weight=class_weight_dict,
        callbacks=[checkpoint, lr_scheduler, early_stopping]
    )
    model.save(model_path)
    np.save(history_path, history.history)
else:
    model = models.load_model(model_path, custom_objects={
        'Patches': Patches,
        'PatchEncoder': PatchEncoder,
    })
    print("Loaded trained model from", model_path)
    history = dict(np.load(history_path, allow_pickle=True).item()) if os.path.exists(history_path) else None

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Unique classes in y_true:", np.unique(y_true))
print("Unique classes in y_pred_classes:", np.unique(y_pred_classes))

results = {
    "accuracy": accuracy_score(y_true, y_pred_classes),
    "classification_report": classification_report(
        y_true, y_pred_classes,
        labels=[0, 1],
        target_names=['REAL', 'FAKE'],
        output_dict=True,
        zero_division=0
    ),
    "confusion_matrix": confusion_matrix(y_true, y_pred_classes, labels=[0,1])
}
np.save(os.path.join(WORK_DIR, "test_results_vit.npy"), results)
np.save(os.path.join(WORK_DIR, "y_test_pred_vit.npy"), y_pred_classes)
np.save(os.path.join(WORK_DIR, "y_test_true_vit.npy"), y_true)

print(f"Test Accuracy: {results['accuracy']*100:.2f}%")

def predict_video(video_path, model, output_size=(128,128), frame_count=10, outpath=None):
    frames = extract_frames(video_path, output_size, frame_count)
    if frames.shape[0] == 0:
        print(f"Error: No frames extracted from {video_path}")
        return
    frames = frames / 255.0
    frames = np.expand_dims(frames, axis=0)
    pred = model.predict(frames)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = float(pred[0][pred_class])
    label = "FAKE" if pred_class == 1 else "REAL"
    outstring = f"{os.path.basename(video_path)}: {label} (Confidence: {confidence:.3f})"
    print("Prediction:", outstring)
    if outpath:
        with open(outpath, "w") as f:
            f.write(outstring)

real_video_files = os.listdir(REAL_PATH)
fake_video_files = os.listdir(FAKE_PATH)

if real_video_files and fake_video_files:
    real_sample = os.path.join(REAL_PATH, real_video_files[0])
    fake_sample = os.path.join(FAKE_PATH, fake_video_files[0])
    predict_video(real_sample, model, outpath=os.path.join(WORK_DIR, "predict_real_vit.txt"))
    predict_video(fake_sample, model, outpath=os.path.join(WORK_DIR, "predict_fake_vit.txt"))

print("All outputs saved in", WORK_DIR)
print(os.listdir(WORK_DIR))

y_test_true = np.load(os.path.join(WORK_DIR, "y_test_true_vit.npy"))
y_test_pred = np.load(os.path.join(WORK_DIR, "y_test_pred_vit.npy"))
cm = confusion_matrix(y_test_true, y_test_pred, labels=[0,1])
report = classification_report(
    y_test_true, y_test_pred,
    labels=[0,1],
    target_names=['REAL', 'FAKE'],
    output_dict=True,
    zero_division=0
)
accuracy = report['accuracy']
f1_score = (report['REAL']['f1-score'] + report['FAKE']['f1-score']) / 2
print(f"Full Test Set Accuracy: {accuracy*100:.2f}%")
print(f"Average F1 Score: {f1_score:.2f}")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix: Deepfake Detection with ViT (Full Test Set)')
plt.show()