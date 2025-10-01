import os
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datetime import datetime
import logging

# Transformers imports
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistilBERTTrainer:
    def __init__(self, data_path: str = "../../data", model_save_path: str = "../../data/models"):
        self.data_path = Path(data_path)
        self.processed_path = self.data_path / "processed"
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self.model_name = "distilbert-base-uncased"
        self.max_length = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_epochs = 3
        self.warmup_steps = 500
        
        logger.info(f"🤖 DistilBERT Trainer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model: {self.model_name}")
    
    def load_data(self):
        """Load processed datasets"""
        logger.info("📊 Loading processed datasets...")
        
        train_df = pd.read_csv(self.processed_path / "sentiment_train.csv")
        val_df = pd.read_csv(self.processed_path / "sentiment_val.csv")
        test_df = pd.read_csv(self.processed_path / "sentiment_test.csv")
        
        logger.info(f"   Train: {len(train_df)} samples")
        logger.info(f"   Validation: {len(val_df)} samples")
        logger.info(f"   Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def create_label_mappings(self, train_df: pd.DataFrame):
        """Create label mappings"""
        unique_labels = sorted(train_df['sentiment_label'].unique())
        label2id = {label: i for i, label in enumerate(unique_labels)}
        id2label = {i: label for label, i in label2id.items()}
        
        logger.info(f"   Labels: {unique_labels}")
        
        return {
            'labels': unique_labels,
            'label2id': label2id,
            'id2label': id2label
        }
    
    def tokenize_data(self, df: pd.DataFrame, tokenizer, label_mappings):
        """Tokenize dataset"""
        
        def tokenize_function(examples):
            # Tokenize text
            tokens = tokenizer(
                examples['clean_text'],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )
            
            # Convert labels to IDs
            tokens['labels'] = [label_mappings['label2id'][label] for label in examples['sentiment_label']]
            return tokens
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def create_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                       tokenizer, label_mappings):
        """Create tokenized datasets"""
        logger.info("🔄 Creating tokenized datasets...")
        
        train_dataset = self.tokenize_data(train_df, tokenizer, label_mappings)
        val_dataset = self.tokenize_data(val_df, tokenizer, label_mappings)
        test_dataset = self.tokenize_data(test_df, tokenizer, label_mappings)
        
        datasets = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        logger.info("✅ Datasets tokenized successfully!")
        return datasets
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
    
    def train_model(self):
        """Complete training pipeline"""
        logger.info("🚀 Starting DistilBERT training pipeline...")
        
        # Load data
        train_df, val_df, test_df = self.load_data()
        
        # Create label mappings
        label_mappings = self.create_label_mappings(train_df)
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create datasets
        datasets = self.create_datasets(train_df, val_df, test_df, tokenizer, label_mappings)
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(label_mappings['labels']),
            id2label=label_mappings['id2label'],
            label2id=label_mappings['label2id']
        )
        model.to(self.device)
        
        # Training arguments - FIXED FOR NEWER TRANSFORMERS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.model_save_path / f"sentiment_training_{timestamp}"
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            warmup_steps=self.warmup_steps,
            weight_decay=0.01,
            logging_dir=str(output_dir / "logs"),
            logging_steps=100,
            eval_strategy="steps",  # FIXED: was evaluation_strategy
            eval_steps=500,
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            report_to=None,
            seed=42,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,  # Added for compatibility
            remove_unused_columns=True,  # Added for compatibility
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Early stopping
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.001
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping]
        )
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"📊 Model parameters: {total_params:,}")
        
        # Train model
        logger.info("🎯 Training model...")
        train_result = trainer.train()
        
        # Evaluate on test set
        logger.info("🧪 Evaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=datasets['test'])
        
        # Save final model
        final_model_path = self.model_save_path / "sentiment"
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        
        # Save label mappings
        with open(final_model_path / "label_mappings.json", "w") as f:
            json.dump(label_mappings, f, indent=2)
        
        # Print final results
        print(f"\n🎉 Training completed!")
        print(f"📈 Final Results:")
        print("="*50)
        print(f"Accuracy: {test_metrics['eval_accuracy']:.4f}")
        print(f"F1 Score (Macro): {test_metrics['eval_f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {test_metrics['eval_f1_weighted']:.4f}")
        
        # Generate detailed classification report
        self.generate_classification_report(datasets['test'], trainer, label_mappings)
        
        logger.info(f"💾 Model saved to: {final_model_path}")
        
        return trainer, test_metrics
    
    def generate_classification_report(self, test_dataset, trainer, label_mappings):
        """Generate detailed classification report"""
        logger.info("📊 Generating classification report...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Convert to label names
        label_names = [label_mappings['id2label'][i] for i in range(len(label_mappings['labels']))]
        
        # Generate report
        report = classification_report(
            y_true, y_pred, 
            target_names=label_names,
            output_dict=True
        )
        
        # Save report
        report_path = self.model_save_path / "sentiment" / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print per-class results
        print(f"\n📊 Per-Class Results:")
        print("-"*30)
        for label in label_names:
            metrics = report[label]
            print(f"{label.title()}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
        
        logger.info(f"📋 Classification report saved to: {report_path}")

def main():
    """Main training function"""
    print("🤖 DistilBERT Sentiment Analysis Training")
    print("="*60)
    
    trainer = DistilBERTTrainer()
    model, metrics = trainer.train_model()
    
    print("\n🎯 Training Pipeline Completed Successfully!")
    print("✅ Model ready for inference!")

if __name__ == "__main__":
    main()
