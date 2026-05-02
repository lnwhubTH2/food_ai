import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer,
    DefaultDataCollator,
    EarlyStoppingCallback
)
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 🌟 ส่วน Global Scope สำหรับ Windows
# ==========================================
model_checkpoint = "google/vit-base-patch16-224-in21k"
# เปิดใช้งาน Fast Image Processor ตามที่เราตกลงกันไว้
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)
accuracy = evaluate.load("accuracy")

def transforms(examples):
    # ปรับภาพและแปลงเป็นตัวเลข
    inputs = image_processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

def compute_metrics(eval_pred):
    # ฟังก์ชันตรวจข้อสอบ
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# ==========================================
# 🚀 ฟังก์ชันหลัก
# ==========================================
def main():
    print("📂 กำลังโหลดชุดข้อมูลจากระบบโฟลเดอร์ (ImageFolder)...")
    
    # 1. ให้ Hugging Face กวาดรูปภาพทั้งหมดใน E:\food_ai\archive\images อัตโนมัติ
    # มันจะเอาชื่อโฟลเดอร์ (เช่น apple_pie, baklava) มาตั้งเป็นชื่อหมวดหมู่ให้เอง
    dataset = load_dataset("imagefolder", data_dir="E:/food_ai/archive/images", split="train")
    
    # 2. แบ่งข้อสอบ: 80% ให้เรียน (Train) | 20% ให้สอบ (Test)
    dataset = dataset.train_test_split(test_size=0.2)

    # 3. ดึงรายชื่อเมนูทั้งหมด 101 เมนูออกมา
    labels = dataset["train"].features["label"].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}
    print(f"✅ ตรวจพบเมนูอาหารทั้งหมด {len(labels)} หมวดหมู่!")

    # 4. นำฟังก์ชันแปลงภาพมาสวมเข้ากับ Dataset
    dataset = dataset.with_transform(transforms)

    print("🧠 กำลังเตรียมสมองกล ViT...")
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True # สั่งลบความจำเดิมทิ้งอย่างสมบูรณ์เพื่อรับ 101 เมนูใหม่
    )

    training_args = TrainingArguments(
        output_dir="./vit-food101-checkpoints",
        remove_unused_columns=False,
        eval_strategy="epoch",      
        save_strategy="epoch",
        
        num_train_epochs=20,              
        learning_rate=3e-5,               
        warmup_ratio=0.1,                 
        lr_scheduler_type="cosine",       
        weight_decay=0.05,                
        
        per_device_train_batch_size=16,   
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,    
        fp16=True,                        
        dataloader_num_workers=4,         
        
        load_best_model_at_end=True,      
        metric_for_best_model="accuracy", 
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )

    print("\n🚀 เริ่มกระบวนการเทรน  (ขั้นตอนนี้อาจจะใช้เวลานานเพราะข้อมูลเยอะมาก)")
    trainer.train()

    print("\n✅ เทรนเสร็จสิ้น! กำลังเซฟโมเดล...")
    trainer.save_model("./my_ultimate_food_vit")
    image_processor.save_pretrained("./my_ultimate_food_vit")
    print("🎉 บันทึกสมองกลก้อนใหม่ไว้ที่โฟลเดอร์ 'my_ultimate_food_vit' เรียบร้อยแล้ว!")

if __name__ == '__main__':
    main()