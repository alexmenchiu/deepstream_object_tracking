import sys
import platform
import configparser
import gi
import cv2
import numpy as np
from openvino.inference_engine import IECore
from gi.repository import GLib, Gst
import pyds
from common.platform_info import PlatformInfo
from common.bus_call import bus_call

gi.require_version('Gst', '1.0')


PGIE_CLASS_ID_VEHICLE = 0
MUXER_BATCH_TIMEOUT_USEC = 33000


ie = IECore()
net = ie.read_network(model='/models/models_re_id/vehicle_reid_0001.xml', weights='/models/models_re_id/vehicle_reid_0001.bin')
exec_net = ie.load_network(network=net, device_name='GPU')
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
input_shape = net.input_info[input_blob].input_data.shape


vehicle_reid_vectors = {}


def vehicle_reid(frame, bbox):
    x, y, w, h = bbox
    vehicle_img = frame[y:y+h, x:x+w]
    vehicle_img = cv2.resize(vehicle_img, (input_shape[3], input_shape[2]))
    vehicle_img = vehicle_img.transpose((2, 0, 1))
    vehicle_img = np.expand_dims(vehicle_img, axis=0)
    vehicle_img = vehicle_img.astype(np.float32) / 255.0
    result = exec_net.infer(inputs={input_blob: vehicle_img})
    reid_vector = result[output_blob][0]  
    return reid_vector


def compare_reid_vectors(vector1, vector2, threshold=0.6):
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return similarity > threshold

def buffer_handler_probe(pad, info, u_data):
    frame_number = 0
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE: 0
    }
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id == PGIE_CLASS_ID_VEHICLE:
                bbox = (int(obj_meta.rect_params.left), int(obj_meta.rect_params.top),
                        int(obj_meta.rect_params.width), int(obj_meta.rect_params.height))
                
                frame_image = np.array(pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id), copy=False, order='C')
                frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2RGB)

                reid_vector = vehicle_reid(frame_image, bbox)

                matched = False
                for object_id, stored_vector in vehicle_reid_vectors.items():
                    if compare_reid_vectors(reid_vector, stored_vector):
                        # print(f"Match found for vehicle ID: {object_id}")
                        matched = True
                        break

                if not matched:
                    new_id = len(vehicle_reid_vectors) + 1
                    vehicle_reid_vectors[new_id] = reid_vector
                    # print(f"New vehicle detected with ID: {new_id}")

            obj_counter[obj_meta.class_id] += 1
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        py_nvosd_text_params.display_text = f"Frame Number={frame_number} Vehicle_count={obj_counter[PGIE_CLASS_ID_VEHICLE]}"
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK



def main(args):
    if len(args) < 2:
        sys.stderr.write("usage: %s <h264_elementary_stream>\n" % args[0])
        sys.exit(1)

    platform_info = PlatformInfo()
    Gst.init(None)

    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    print("Creating Source \n ")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    print("Creating H264Parser \n")
    parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not parser:
        sys.stderr.write(" Unable to create h264 parser \n")

    print("Creating Decoder \n")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    primary_inference = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not primary_inference:
        sys.stderr.write(" Unable to create primary inferene\n")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker\n")

    secondary_inference = Gst.ElementFactory.make("nvinfer", "secondary-inference1")
    if not secondary_inference:
        sys.stderr.write(" Unable to create secondary1 inferene\n")

    converter = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not converter:
        sys.stderr.write(" Unable to create nvvidconv \n")

    displayer = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not displayer:
        sys.stderr.write(" Unable to create nvosd \n")

    if platform_info.is_platform_aarch64():
        transform = None
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        print("Creating nveglglessink \n")
        if platform_info.is_platform_aarch64():
            transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
            if not transform:
                sys.stderr.write(" Unable to create nvegltransform \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create nveglglessink \n")

    source.set_property('location', args[1])
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)

    primary_inference.set_property('config-file-path', "../configuration_files/primary_detection_configuration.txt")
    secondary_inference.set_property('config-file-path', "../configuration_files/secondary_detection_configuration.txt")

    config = configparser.ConfigParser()
    config.read('../configuration_files/tracking_configuration.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(primary_inference)
    pipeline.add(tracker)
    pipeline.add(secondary_inference)
    pipeline.add(converter)
    pipeline.add(displayer)
    if platform_info.is_platform_aarch64() and not platform_info.is_integrated_gpu():
        pipeline.add(transform)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    source.link(parser)
    parser.link(decoder)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    srcpad.link(sinkpad)
    streammux.link(primary_inference)
    primary_inference.link(tracker)
    tracker.link(secondary_inference)
    secondary_inference.link(converter)
    converter.link(displayer)
    if platform_info.is_platform_aarch64() and not platform_info.is_integrated_gpu():
        displayer.link(transform)
        transform.link(sink)
    else:
        displayer.link(sink)

    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    osdsinkpad = displayer.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, buffer_handler_probe, 0)


    print("Starting pipeline \n")
    
    pipeline.set_state(Gst.State.PLAYING)
    try:
      loop.run()
    except:
      pass

    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
