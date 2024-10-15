mod bbox;
mod get_face_onnx;
mod get_face_torch;
mod get_face_tract;
mod inference_model;
use inference_model::{get_bbox, Inference, OnnxModel, TorchModel, TractModel};


fn main() {
    let _onnx = OnnxModel::load("models/yolov8n_face.onnx", false).unwrap();
    let _torch = TorchModel::load("models/yolov8n-face.torchscript", false).unwrap();
    let _tract = TractModel::load("models/yolov8n_face.onnx", false).unwrap();
    let test_img = image::open("/home/hbdesk/Downloads/dog.jpeg").unwrap();
    let mut onnx_time: Vec<u128> = vec![];
    let mut tract_time: Vec<u128> = vec![];
    let mut torch_time: Vec<u128> = vec![];

    for _ in 0..20 {
        let t1 = std::time::Instant::now();
        let _ = get_bbox(&_onnx, &test_img, 0.5,0.5);
        onnx_time.push(t1.elapsed().as_millis()); 

        let t2 = std::time::Instant::now();
        let _2 = get_bbox(&_tract, &test_img, 0.5,0.5);
        tract_time.push(t2.elapsed().as_millis()); 

        let t3 = std::time::Instant::now();
        let _3 = get_bbox(&_torch, &test_img, 0.5,0.5);
        torch_time.push(t3.elapsed().as_millis()); 
    }
    let _a = average(onnx_time); 
    let _b = average(tract_time); 
    let _c = average(torch_time); 
    println!("onnx_time: {:?}\ntract_time: {:?}\ntorch_time: {:?}", _a, _b, _c); 
}


fn average(numbers: Vec<u128>) -> f32 {
    let nnumbers = numbers.len() as f32;
    let mut sum: f32 = 0.0;
    for n in numbers {
        sum += (n as f32);
    }
    sum / nnumbers
}



