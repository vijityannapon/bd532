# Section 5 — Conclusion, Reflection & Path to Independent Study

## 5.1 ผลลัพธ์เทียบกับเป้าหมาย

| เป้าหมาย (Section 2.1) | เกณฑ์ | ผลที่ได้ | สถานะ |
|---|---|---|---|
| Model: แยกผู้กู้ดีและไม่ดี | ROC-AUC ≥ 0.75 | **0.764** (test), CV 0.756 ± 0.005 | ✅ |
| Model: มาตรฐาน credit scoring | KS ≥ 0.35 | **0.399** | ✅ |
| Model: PD เชื่อถือได้ | Brier ≤ baseline | 0.068 (baseline LogReg 0.069) และ calibration curve ใกล้เส้นทแยง | ✅ |
| User (ผู้กู้): เป็นธรรม | Disparate Impact ≥ 0.8 | เพศ **0.956**, อายุ **0.829** | ✅ (อายุเฉียดเกณฑ์) |
| Model: ช่องว่าง FPR ระหว่างเพศ | ≤ 0.05 | 0.016 (val) / 0.028 (test) | ✅ |
| System: ความเร็ว | p95 < 300 ms | ~40 ms ต่อ request รวม SHAP | ✅ |
| System: ตรวจสอบได้ | log ทุก prediction + version | request_id, model version, MLflow run id ทุก response | ✅ |
| User (เจ้าหน้าที่): เหตุผลที่เข้าใจได้ | มีเหตุผลทุกเคส | top-k SHAP เป็นภาษาไทย ▲/▼ พร้อมค่าจริง | ✅ |
| Org: ลด FPD / เวลาพิจารณา | −15% / ≤ 1 วัน | **วัดไม่ได้ใน PoC** ต้องทำ pilot จริง | 🔜 |

**ผลกระทบที่คาดได้จาก test set:** ถ้าอนุมัติเร็วเฉพาะกลุ่ม LOW_RISK (68% ของคำขอ) กลุ่มนี้มีอัตราผิดนัด 3.7% ต่ำกว่าค่าเฉลี่ยพอร์ต 8.1% อยู่ 54% ส่วนเจ้าหน้าที่ใช้เวลาไปกับ 32% ของคำขอที่เสี่ยงจริง

## 5.2 Critical Reflection

**สิ่งที่ได้ผลดี**
* **ให้ data เป็นตัวตัดสิน:** บนข้อมูลสังเคราะห์ LogReg ชนะ แต่บนข้อมูลจริง boosting ชนะ จึงไม่ควรเลือกโมเดลจากสมมติฐาน
* **เกณฑ์เลือกโมเดลหลายมิติ:** calibration gate ตัด Random Forest ที่ AUC ไม่แย่แต่ PD ผิดเพี้ยน (ถ้าดูแค่ AUC จะพลาด) และเมื่อ AUC เท่ากัน fairness ช่วยตัดสินจน gender FPR gap ลดลง 63% โดยไม่เสียความแม่นยำ
* **Human-in-the-loop by design:** FTA ไม่มี single-point failure เพราะทุก MCS ต้องมีความผิดพลาดทั้งของโมเดลและของมนุษย์

**ข้อจำกัดและสิ่งที่ยังไม่แน่ใจ**
* **Selection bias (ENV-5):** มี label เฉพาะคนที่เคยได้รับอนุมัติ โมเดลจึงไม่เคยเห็นผลของคนที่ถูกปฏิเสธ
* **ใช้ข้อมูลเพียงตารางเดียว:** Home Credit มีตาราง bureau, previous applications และ installments ด้วย ซึ่งงานอื่นรายงานว่าดัน AUC ได้ถึง ~0.79 (PoC นี้จงใจจำกัดเพื่อให้ UI กรอกข้อมูลได้จริง)
* **Cost ratio 5:1 เป็นสมมติฐาน:** threshold และ band ขึ้นกับค่านี้โดยตรง ควรคำนวณจาก LGD และ margin จริงของธนาคาร
* **Proxy ที่ยังเหลือ:** `own_car` สำคัญอันดับ 4 และสัมพันธ์กับเพศ ส่วน reweighing แก้ได้เฉพาะเพศ ไม่แก้ช่องว่างตามอายุ
* **SHAP ไม่ใช่ causal:** เหตุผลอย่าง “อายุ 24 ลดความเสี่ยง” (interaction) อาจทำให้เจ้าหน้าที่สับสน การอธิบายต้องมาพร้อม training ผู้ใช้
* **PoC engineering:** log เป็นไฟล์ JSONL, ไม่มี authentication และยังไม่ได้วัด latency ภายใต้ load จริง

## 5.3 ต่อยอดเป็น Independent Study

| ระยะ | งาน | คำถามวิจัย |
|---|---|---|
| 1 | เพิ่ม feature จาก bureau / previous applications / installments | ข้อมูลพฤติกรรมเพิ่ม AUC ได้เท่าไร และเปลี่ยน fairness อย่างไร |
| 2 | เปรียบเทียบวิธีลด bias: reweighing, constraint-based (Exponentiated Gradient), threshold optimization | Trade-off ระหว่าง accuracy, fairness และความชอบด้วยกฎหมาย (equal treatment) |
| 3 | Reject inference / semi-supervised เพื่อลด selection bias | ประเมินผู้กู้ที่ไม่เคยได้รับอนุมัติได้ดีขึ้นแค่ไหน |
| 4 | Drift monitoring (PSI) + retraining policy จาก feedback loop | ควร retrain เมื่อไร และ feedback ของเจ้าหน้าที่ช่วยโมเดลจริงหรือไม่ |
| 5 | User study กับเจ้าหน้าที่สินเชื่อ: เทียบมีและไม่มีคำอธิบาย SHAP | คำอธิบายเพิ่มความถูกต้องของการตัดสินใจ หรือเพิ่ม automation bias |
