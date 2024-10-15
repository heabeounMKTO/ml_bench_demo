mod bbox;
mod get_face_onnx;
// mod get_face_torch;
mod get_face_tract;



fn main() {

    let load_mod = get_face_tract::load_model("models/yolov8n_face.onnx");
}
