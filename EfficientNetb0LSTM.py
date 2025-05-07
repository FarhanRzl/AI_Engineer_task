import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time
import sys

class HumanActionRecognitionSystem:
    def __init__(self):
        self.dataset_path = ""
        self.feature_path = "extracted_features"
        self.model_path = "model_checkpoints"
        self.class_names = []

    def display_header(self):
        """Display header for the application"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 80)
        print(" " * 25 + "HUMAN ACTION RECOGNITION SYSTEM")
        print("=" * 80)
        print(" Developed with TensorFlow and EfficientNet")
        print("-" * 80)

    def display_menu(self):
        """Display main menu of the application"""
        print("\nMAIN MENU:")
        print("1. Set Dataset Path")
        print("2. Process Dataset")
        print("3. Train Model")
        print("4. Resume Training From Checkpoint")
        print("5. Predict from Video")
        print("6. Generate Label File from Videos")
        print("7. About System")
        print("8. Exit")

        choice = input("\nEnter your choice (1-8): ")
        return choice

    def set_dataset_path(self):
        """Set dataset path"""
        self.display_header()
        print("\n[SET DATASET PATH]")
        print("-" * 80)

        self.dataset_path = input("Enter video dataset path: ")

        if not os.path.exists(self.dataset_path):
            print(f"\n[ERROR] Directory not found at {self.dataset_path}")
            alternative_dir = input("Enter alternative path (or press Enter to return): ")
            if not alternative_dir:
                print("\nReturning to main menu...")
                time.sleep(1)
                return False
            if not os.path.exists(alternative_dir):
                print(f"\n[ERROR] Alternative directory not found at {alternative_dir}")
                print("\nReturning to main menu...")
                time.sleep(2)
                return False
            self.dataset_path = alternative_dir

        # Validate class names
        class_dirs = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        if not class_dirs:
            print(f"\n[ERROR] No class directories found in {self.dataset_path}")
            print("Make sure your dataset is organized with one folder per action class.")
            input("\nPress Enter to continue...")
            return False

        self.class_names = class_dirs
        print(f"\n[SUCCESS] Dataset path set to: {self.dataset_path}")
        print(f"Found {len(self.class_names)} action classes: {', '.join(self.class_names)}")
        input("\nPress Enter to continue...")
        return True

    def process_dataset(self):
        """Process dataset in chunks"""
        self.display_header()
        print("\n[PROCESS DATASET]")
        print("-" * 80)

        if not self.dataset_path:
            print("[ERROR] Dataset path not set. Please set dataset path first.")
            input("\nPress Enter to continue...")
            return

        # Create directories if they don't exist
        os.makedirs(self.feature_path, exist_ok=True)

        # Display processing options
        print(f"Dataset path: {self.dataset_path}")
        print(f"Feature path: {self.feature_path}")
        print(f"Found {len(self.class_names)} action classes")

        # Ask for processing parameters
        image_size = input("\nEnter image size (default: 160x160): ") or "160x160"
        width, height = map(int, image_size.split('x'))

        sequence_length = int(input("Enter sequence length (default: 20): ") or "20")
        chunk_size = int(input("Enter chunk size (default: 5): ") or "5")

        print("\nStarting dataset processing in chunks...")
        print("-" * 50)

        # Show progress bar for each class
        for i in range(0, len(self.class_names), chunk_size):
            chunk_classes = self.class_names[i:i+chunk_size]
            chunk_dir = os.path.join(self.feature_path, f"chunk_{i}")
            os.makedirs(chunk_dir, exist_ok=True)

            print(f"\nProcessing chunk {i//chunk_size + 1}/{(len(self.class_names)-1)//chunk_size + 1}: {chunk_classes}")

            for class_name in chunk_classes:
                print(f"  Processing class: {class_name}")
                # Process single class
                result = self.process_single_class(
                    class_name,
                    self.dataset_path,
                    chunk_dir,
                    image_size=(width, height),
                    sequence_length=sequence_length
                )

                if result:
                    features, labels = result
                    print(f"  ✓ Extracted {features.shape[0]} samples for class {class_name}")
                else:
                    print(f"  ✗ Failed to extract features for class {class_name}")

                # Clear TensorFlow session to free memory
                tf.keras.backend.clear_session()

        # Combine all features
        print("\nCombining features from all chunks...")
        self.combine_features_from_chunks(self.feature_path)

        print("\n[SUCCESS] Dataset processing completed!")
        input("\nPress Enter to continue...")

    def train_model(self):
        """Train model with class balancing"""
        self.display_header()
        print("\n[TRAIN MODEL]")
        print("-" * 80)

        if not os.path.exists(self.feature_path):
            print("[ERROR] Feature path not found. Please process dataset first.")
            input("\nPress Enter to continue...")
            return

        # Check if features exist
        feature_files = [f for f in os.listdir(self.feature_path) if f.startswith('features_') and f.endswith('.npz')]
        if not feature_files:
            print("[ERROR] No feature files found. Please process dataset first.")
            input("\nPress Enter to continue...")
            return

        # Ask for training parameters
        batch_size = int(input("Enter batch size (default: 4): ") or "4")
        epochs = int(input("Enter number of epochs (default: 50): ") or "50")

        # Confirm training
        print(f"\nReady to train model with:")
        print(f"- {len(feature_files)} feature files")
        print(f"- Batch size: {batch_size}")
        print(f"- Epochs: {epochs}")
        confirm = input("\nStart training? (y/n): ").lower()

        if confirm != 'y':
            print("Training cancelled.")
            input("\nPress Enter to continue...")
            return

        print("\nStarting model training with class balancing...")
        print("-" * 50)

        self.train_with_class_balancing(
            feature_path=self.feature_path,
            model_path=self.model_path,
            batch_size=batch_size,
            epochs=epochs
        )

        print("\n[SUCCESS] Model training completed!")
        input("\nPress Enter to continue...")

    def resume_training(self):
        """Resume training from checkpoint"""
        self.display_header()
        print("\n[RESUME TRAINING FROM CHECKPOINT]")
        print("-" * 80)

        # Check if model checkpoint exists
        checkpoint_path = os.path.join(self.model_path, 'best_model.h5')
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
            input("\nPress Enter to continue...")
            return

        # Load data for training
        print("Loading data for training...")
        features, labels, class_data = self.load_balanced_data(self.feature_path)

        if features is None or labels is None:
            print("[ERROR] Failed to load data. Please check feature extraction.")
            input("\nPress Enter to continue...")
            return

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        print(f"Training data: {len(X_train)} samples")
        print(f"Validation data: {len(X_val)} samples")
        print(f"Testing data: {len(X_test)} samples")

        # Ask for training parameters
        epochs = int(input("\nEnter number of epochs (default: 50): ") or "50")
        batch_size = int(input("Enter batch size (default: 4): ") or "4")

        # Confirm resuming
        confirm = input("\nResume training from checkpoint? (y/n): ").lower()
        if confirm != 'y':
            print("Training cancelled.")
            input("\nPress Enter to continue...")
            return

        print("\nResuming training from checkpoint...")

        model, history = self.resume_training_from_checkpoint(
            checkpoint_path,
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )

        if model is not None:
            print("\n[SUCCESS] Training resumed and completed!")
        else:
            print("\n[ERROR] Failed to resume training.")

        input("\nPress Enter to continue...")

    def predict_from_video(self):
        """Predict action from a single video"""
        self.display_header()
        print("\n[PREDICT FROM VIDEO]")
        print("-" * 80)

        # Check if model exists
        model_path = os.path.join(self.model_path, 'best_model.h5')
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found at {model_path}")
            model_path = input("Enter path to model file (or press Enter to return): ")
            if not model_path or not os.path.exists(model_path):
                print("Prediction cancelled.")
                input("\nPress Enter to continue...")
                return

        # Check if class names file exists
        class_names_path = os.path.join(self.model_path, 'class_names.txt')
        if not os.path.exists(class_names_path):
            print(f"[ERROR] Class names file not found at {class_names_path}")
            print("Using detected class names from dataset.")

            # Create class names file
            with open(class_names_path, 'w') as f:
                for class_name in self.class_names:
                    f.write(f"{class_name}\n")

        # Get video path
        video_path = input("\nEnter path to video file: ")
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found at {video_path}")
            input("\nPress Enter to continue...")
            return

        # Ask for prediction parameters
        image_size = input("\nEnter image size (default: 160x160): ") or "160x160"
        width, height = map(int, image_size.split('x'))

        sequence_length = int(input("Enter sequence length (default: 20): ") or "20")

        print("\nProcessing video and making prediction...")

        # Make prediction
        predicted_class, confidence, top_predictions = self.predict_video(
            video_path,
            model_path,
            class_names_path,
            image_size=(width, height),
            sequence_length=sequence_length
        )

        print("\n" + "=" * 50)
        print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
        print("-" * 50)
        print("Top 3 predictions:")
        for i, (class_name, prob) in enumerate(top_predictions):
            print(f"{i+1}. {class_name}: {prob:.4f}")
        print("=" * 50)

        input("\nPress Enter to continue...")

    def generate_label_file(self):
        """Generate label file from videos"""
        self.display_header()
        print("\n[GENERATE LABEL FILE]")
        print("-" * 80)

        if not self.dataset_path:
            print("[ERROR] Dataset path not set. Please set dataset path first.")
            input("\nPress Enter to continue...")
            return

        # Check if class directories exist
        class_dirs = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        if not class_dirs:
            print(f"[ERROR] No class directories found in {self.dataset_path}")
            input("\nPress Enter to continue...")
            return

        # Show available classes
        print("Available classes:")
        for i, class_name in enumerate(class_dirs):
            print(f"{i+1}. {class_name}")

        # Ask for output file
        output_file = input("\nEnter path for label file (default: 'video_labels.txt'): ") or 'video_labels.txt'

        # Create label mapping
        label_mapping = {}
        for i, class_name in enumerate(class_dirs):
            use_class = input(f"\nInclude class '{class_name}'? (y/n, default: y): ") or 'y'
            if use_class.lower() == 'y':
                label_mapping[class_name] = i

        if not label_mapping:
            print("[ERROR] No classes selected.")
            input("\nPress Enter to continue...")
            return

        # Generate label file
        print("\nGenerating label file...")
        video_count = self.create_label_file(self.dataset_path, output_file, label_mapping)

        if video_count > 0:
            print(f"\n[SUCCESS] Created label file with {video_count} videos at {output_file}")
        else:
            print("\n[ERROR] No videos found for label file generation.")

        input("\nPress Enter to continue...")

    def about_system(self):
        """Display information about the system"""
        self.display_header()
        print("\n[ABOUT SYSTEM]")
        print("-" * 80)
        print("\nHuman Action Recognition System")
        print("Version 1.0")
        print("\nThis system uses deep learning to recognize human actions in videos.")
        print("It employs EfficientNetB0 for feature extraction and LSTM for sequence modeling.")
        print("\nFeatures:")
        print("- Memory-efficient processing for large datasets")
        print("- Class-by-class feature extraction")
        print("- Resumable training from checkpoints")
        print("- Prediction from single videos")

        input("\nPress Enter to continue...")

    def run(self):
        """Run the main application loop"""
        while True:
            self.display_header()
            choice = self.display_menu()

            if choice == '1':
                self.set_dataset_path()
            elif choice == '2':
                self.process_dataset()
            elif choice == '3':
                self.train_model()
            elif choice == '4':
                self.resume_training()
            elif choice == '5':
                self.predict_from_video()
            elif choice == '6':
                self.generate_label_file()
            elif choice == '7':
                self.about_system()
            elif choice == '8':
                self.display_header()
                print("\nThank you for using the Human Action Recognition System!")
                print("Exiting...")
                sys.exit(0)
            else:
                print("\n[ERROR] Invalid choice. Please try again.")
                time.sleep(1)

    def process_single_class(self, class_name, dataset_path, feature_path,
                            image_size=(160, 160), sequence_length=20,
                            batch_size=2):
        """
        Process videos for one class separately.
        This strategy allows processing large datasets with limited memory.
        """
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory for class {class_name} not found at {class_dir}")
            return None, None

        # Set up feature extractor
        feature_extractor = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(image_size[0], image_size[1], 3)
        )

        # Freeze EfficientNet layers
        for layer in feature_extractor.layers:
            layer.trainable = False

        video_paths = []
        labels = []

        # Get all videos for this class
        for filename in os.listdir(class_dir):
            if filename.endswith(('.avi', '.mp4')):
                video_path = os.path.join(class_dir, filename)
                video_paths.append(video_path)
                # Class index will be added later
                labels.append(0)  # Temporary placeholder

        # If no videos, return
        if not video_paths:
            print(f"No videos found for class {class_name}")
            return None, None

        # Process videos batch by batch
        all_features = []
        batch_video_paths = []

        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i:i+batch_size]
            batch_video_paths.extend(batch_paths)

            # Process batch of videos
            X_batch = []
            for video_path in batch_paths:
                # Extract frames from video
                frames = self.get_frames_from_video(video_path, image_size, sequence_length)

                if not frames:
                    print(f"Warning: No frames extracted from {video_path}")
                    continue

                # Normalize frames
                normalized_frames = self.normalize_frames(frames)
                X_batch.append(normalized_frames)

            if not X_batch:
                continue

            # Convert to numpy array
            X_batch = np.array(X_batch)

            # Extract features
            batch_features = []

            # Process each video separately to save memory
            for j in range(len(X_batch)):
                video_frames = X_batch[j]
                video_features = []

                # Process frames in batches
                for k in range(0, len(video_frames), batch_size):
                    end_idx = min(k + batch_size, len(video_frames))
                    frame_batch = video_frames[k:end_idx]
                    frame_features = feature_extractor.predict(frame_batch, verbose=0)
                    video_features.append(frame_features)

                # Combine all frame features
                video_features = np.vstack(video_features)
                batch_features.append(video_features)

            # Add to all_features
            all_features.extend(batch_features)

            # Clean up memory
            del X_batch, batch_features
            tf.keras.backend.clear_session()

        # Convert to numpy array
        if all_features:
            all_features = np.array(all_features)
            labels = np.zeros(len(all_features))  # All labels the same (one class)

            # Save features for this class
            os.makedirs(feature_path, exist_ok=True)
            output_file = os.path.join(feature_path, f"features_{class_name}.npz")
            np.savez_compressed(output_file, features=all_features, labels=labels, paths=batch_video_paths)

            return all_features, labels
        else:
            return None, None

    def resume_training_from_checkpoint(self, checkpoint_path, X_train, y_train, X_val, y_val,
                                    epochs=50, batch_size=4):
        """
        Resume training from a saved checkpoint.
        Useful if training process was interrupted.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return None, None

        print(f"Resuming training from {checkpoint_path}")

        # Load model
        model = load_model(checkpoint_path)

        # Determine initial epoch
        initial_epoch = 0

        # Look for history file if it exists
        history_file = os.path.join(os.path.dirname(checkpoint_path), "history.npz")
        if os.path.exists(history_file):
            history_data = np.load(history_file, allow_pickle=True)
            history_dict = history_data['history'].item()
            initial_epoch = len(history_dict.get('loss', []))
            print(f"Resuming from epoch {initial_epoch}")

        # Callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        # Callback to save history
        class SaveHistoryCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                # Update history
                if not hasattr(self, 'history'):
                    self.history = {k: [] for k in logs.keys()}

                for k, v in logs.items():
                    self.history[k].append(v)

                # Save history
                np.savez_compressed(history_file, history=self.history)

        save_history = SaveHistoryCallback()

        # Continue training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            initial_epoch=initial_epoch,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping, reduce_lr, save_history]
        )

        return model, history

    def combine_features_from_chunks(self, feature_path):
        """
        Combine all features from the processed chunks.
        """
        all_feature_files = []

        # Find all npz files in subdirectories
        for root, dirs, files in os.walk(feature_path):
            for file in files:
                if file.endswith('.npz') and file.startswith('features_'):
                    all_feature_files.append(os.path.join(root, file))

        print(f"Found {len(all_feature_files)} feature files across all chunks")

        # Copy all files to main directory
        copied_count = 0
        for file_path in all_feature_files:
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(feature_path, file_name)

            if os.path.abspath(file_path) != os.path.abspath(dest_path):
                try:
                    shutil.copy2(file_path, dest_path)
                    copied_count += 1
                    print(f"Copied {file_path} to {dest_path}")
                except Exception as e:
                    print(f"Error copying {file_path}: {e}")

        print(f"Successfully copied {copied_count} feature files to the main directory")

    def get_frames_from_video(self, video_path, target_size, max_frames=None):
        """
        Extract frames from video with memory constraints.
        """
        import cv2
        frames = []
        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # If frames more than max_frames, take frames evenly
        if max_frames and frame_count > max_frames:
            indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
            current_frame = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame in indices:
                    # Resize frame
                    frame = cv2.resize(frame, target_size)
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

                current_frame += 1
        else:
            # If frames less than max_frames, take all
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame
                frame = cv2.resize(frame, target_size)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        # If frames less than max_frames, pad with last frame
        if max_frames and len(frames) < max_frames and frames:
            last_frame = frames[-1]
            padding = [last_frame] * (max_frames - len(frames))
            frames.extend(padding)

        return frames

    def normalize_frames(self, frames):
        """
        Normalize frames for EfficientNet input.
        """
        frames = np.array(frames, dtype=np.float32)
        normalized_frames = tf.keras.applications.efficientnet.preprocess_input(frames)
        return normalized_frames

    def train_with_class_balancing(self, feature_path, model_path, batch_size=4, epochs=50):
        """
        Train model with balanced class samples.
        """
        # Get all feature files
        feature_files = [f for f in os.listdir(feature_path) if f.startswith('features_') and f.endswith('.npz')]

        if not feature_files:
            print("No feature files found!")
            return

        # Load data per class and find minimum samples
        min_samples = float('inf')
        class_data = {}

        for file in feature_files:
            data = np.load(os.path.join(feature_path, file))
            features = data['features']

            class_name = file.replace('features_', '').replace('.npz', '')
            class_data[class_name] = features

            min_samples = min(min_samples, len(features))

        print(f"Minimum samples per class: {min_samples}")

        # Take equal samples from each class
        balanced_features = []
        balanced_labels = []

        for idx, (class_name, features) in enumerate(class_data.items()):
            # Take random samples
            indices = np.random.choice(len(features), min_samples, replace=False)
            balanced_features.append(features[indices])
            balanced_labels.append(np.ones(min_samples, dtype=int) * idx)

        # Combine all data
        X = np.concatenate(balanced_features, axis=0)
        y = np.concatenate(balanced_labels, axis=0)

        # Split for training/validation/testing
        from sklearn.model_selection import train_test_split

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Testing: {len(X_test)}")

        # Input shape and number of classes
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(class_data)

        # Build model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape, recurrent_dropout=0.2),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        os.makedirs(model_path, exist_ok=True)
        checkpoint_path = os.path.join(model_path, 'best_model.h5')

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        # Train model with generator to save memory
        def data_generator(features, labels, batch_size):
            num_samples = len(features)
            indices = np.arange(num_samples)

            while True:
                np.random.shuffle(indices)
                for i in range(0, num_samples, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    yield features[batch_indices], labels[batch_indices]

        train_gen = data_generator(X_train, y_train, batch_size)
        val_gen = data_generator(X_val, y_val, batch_size)

        # Steps per epoch
        steps_per_epoch = max(1, len(X_train) // batch_size)
        validation_steps = max(1, len(X_val) // batch_size)

        # Train model
        try:
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                epochs=epochs,
                callbacks=[checkpoint, early_stopping, reduce_lr]
            )
        except KeyboardInterrupt:
            print("Training interrupted by user.")

        # Save history
        history_file = os.path.join(model_path, "history.npz")
        np.savez_compressed(history_file, history=history.history)

        # Save class names
        class_names_path = os.path.join(model_path, "class_names.txt")
        with open(class_names_path, "w") as f:
            for class_name in class_data.keys():
                f.write(f"{class_name}\n")

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")

        return model

    def load_balanced_data(self, feature_path):
        """
        Load data with balanced classes for training.
        """
        # Get all feature files
        feature_files = [f for f in os.listdir(feature_path) if f.startswith('features_') and f.endswith('.npz')]

        if not feature_files:
            print("No feature files found!")
            return None, None, None

        # Load data per class and find minimum samples
        min_samples = float('inf')
        class_data = {}

        for file in feature_files:
            try:
                data = np.load(os.path.join(feature_path, file))
                features = data['features']

                class_name = file.replace('features_', '').replace('.npz', '')
                class_data[class_name] = features

                min_samples = min(min_samples, len(features))
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if not class_data:
            return None, None, None

        print(f"Minimum samples per class: {min_samples}")

        # Take equal samples from each class
        balanced_features = []
        balanced_labels = []

        for idx, (class_name, features) in enumerate(class_data.items()):
            # Take random samples
            indices = np.random.choice(len(features), min_samples, replace=False)
            balanced_features.append(features[indices])
            balanced_labels.append(np.ones(min_samples, dtype=int) * idx)

        # Combine all data
        X = np.concatenate(balanced_features, axis=0)
        y = np.concatenate(balanced_labels, axis=0)

        return X, y, class_data

    def predict_video(self, video_path, model_path, class_names_path,
                    image_size=(160, 160), sequence_length=20):
        """
        Predict action from a video.
        """
        import cv2

        # Check if files exist
        if not os.path.exists(video_path):
            print(f"Video file not found at {video_path}")
            return None, 0, []

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return None, 0, []

        if not os.path.exists(class_names_path):
            print(f"Class names file not found at {class_names_path}")
            return None, 0, []

        # Load class names
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

        # Load the model
        model = load_model(model_path)

        # Set up feature extractor
        feature_extractor = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(image_size[0], image_size[1], 3)
        )

        # Extract frames from video
        frames = self.get_frames_from_video(video_path, image_size, sequence_length)

        if not frames:
            print(f"No frames extracted from {video_path}")
            return None, 0, []

        # Normalize frames
        normalized_frames = self.normalize_frames(frames)

        # Extract features for each frame
        features = []
        for frame in normalized_frames:
            frame_features = feature_extractor.predict(np.expand_dims(frame, axis=0), verbose=0)
            features.append(frame_features[0])

        features = np.array([features])

        # Make prediction
        predictions = model.predict(features)[0]
        predicted_class_index = np.argmax(predictions)
        confidence = predictions[predicted_class_index]

        # Get class name
        predicted_class = class_names[predicted_class_index]

        # Get top 3 predictions
        top_indices = np.argsort(predictions)[::-1][:3]
        top_predictions = [(class_names[i], predictions[i]) for i in top_indices]

        return predicted_class, confidence, top_predictions

    def create_label_file(self, dataset_path, output_file, label_mapping):
        """
        Create a label file for all videos in the dataset.
        Format: video_path class_index
        """
        with open(output_file, 'w') as f:
            video_count = 0

            for class_name, class_index in label_mapping.items():
                class_dir = os.path.join(dataset_path, class_name)

                if not os.path.isdir(class_dir):
                    continue

                for filename in os.listdir(class_dir):
                    if filename.endswith(('.avi', '.mp4')):
                        video_path = os.path.join(class_dir, filename)
                        f.write(f"{video_path} {class_index}\n")
                        video_count += 1

            return video_count

if __name__ == "__main__":
    system = HumanActionRecognitionSystem()
    system.run()