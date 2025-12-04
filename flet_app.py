import flet as ft
import cv2
import numpy as np
import base64
import os
import sys
import threading
from typing import Optional, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.insightface_pipeline import InsightFacePipeline
from inference.pipeline import FaceRecognitionPipeline

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='MagFace Face Recognition System')
parser.add_argument('--model', type=str, default='insightface', choices=['insightface', 'magface'],
                    help='Model type to use: insightface or magface')
args, unknown = parser.parse_known_args()
SELECTED_MODEL = args.model

def cv2_to_base64(img_arr):
    """Convert an OpenCV image to a Base64 string for Flet display"""
    if img_arr is None:
        return None
    try:
        # 确保是 RGB
        if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
            pass

        _, buffer = cv2.imencode('.jpg', img_arr)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return b64_str
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def main(page: ft.Page):
    page.title = "MagFace Face Recognition System"
    page.theme_mode = ft.ThemeMode.LIGHT  # Default to a white theme
    page.bgcolor = ft.Colors.WHITE  # Set white background
    page.padding = 0

    # Window settings - enable resizing (Flet 0.28+ API)
    page.window.width = 1200
    page.window.height = 800
    page.window.min_width = 800  # Minimum width
    page.window.min_height = 600  # Minimum height
    page.window.resizable = True  # Allow window resizing
    page.window.maximizable = True  # Allow maximization

    # --- Global state ---
    pipeline: Optional[Union[InsightFacePipeline, FaceRecognitionPipeline]] = None

    def get_db_path(model_type):
        return f'database/{model_type}_face_database.pkl'

    # State variables
    use_gpu = True
    quality_threshold = 0.5
    similarity_threshold = 0.4
    current_model_type = SELECTED_MODEL

    db_paths = {
        'insightface': get_db_path('insightface'),
        'magface': get_db_path('magface')
    }
    db_path_field = None  # 占位，稍后初始化 UI 控件

    def get_current_db_path() -> str:
        nonlocal db_paths
        field_value = (db_path_field.value.strip() if db_path_field and db_path_field.value else '')
        if field_value:
            db_paths[current_model_type] = field_value
            return field_value
        return db_paths.get(current_model_type) or get_db_path(current_model_type)

    def set_db_path_value(path: str, update_field: bool = True):
        if not path:
            return
        db_paths[current_model_type] = path
        if db_path_field and update_field:
            db_path_field.value = path
            db_path_field.update()

    def on_db_path_change(e):
        value = (db_path_field.value or '').strip() if db_path_field else ''
        if value:
            db_paths[current_model_type] = value

    def on_db_path_selected(e: ft.FilePickerResultEvent):
        if e.files:
            selected = e.files[0].path
            set_db_path_value(selected)
        elif e.path:
            set_db_path_value(e.path)

    # --- UI component references ---
    status_text = ft.Text("Ready", size=12)
    db_path_field = ft.TextField(
        label="Database path",
        value=db_paths[current_model_type],
        expand=True,
        dense=True,
        on_change=on_db_path_change,
        helper_text="Pick or type the .pkl file used for save/load"
    )

    # Registration tab widgets
    reg_name_field = ft.TextField(
        label="Name", width=300,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.WHITE),
        border_color=ft.Colors.WHITE,
        text_style=ft.TextStyle(color=ft.Colors.WHITE),
        label_style=ft.TextStyle(color=ft.Colors.WHITE70),
        cursor_color=ft.Colors.WHITE
    )
    reg_img_control = ft.Image(src="", width=400, height=400, fit=ft.ImageFit.CONTAIN, visible=False, border_radius=20)
    reg_img_container = ft.Container(
        content=reg_img_control,
        on_click=lambda _: pick_reg_image_dialog.pick_files(allow_multiple=False)
    )
    reg_img_placeholder = ft.Container(
        content=ft.Column([
            ft.Icon(ft.Icons.CAMERA_ALT, size=60, color=ft.Colors.WHITE70),
            ft.Text("Click to upload", color=ft.Colors.WHITE70, size=16)
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=400, height=400,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLACK),
        border=ft.border.all(2, ft.Colors.with_opacity(0.3, ft.Colors.WHITE)),
        border_radius=20,
        alignment=ft.alignment.center,
        on_click=lambda _: pick_reg_image_dialog.pick_files(allow_multiple=False)
    )
    reg_img_path = None

    # Recognition tab widgets
    rec_img_control = ft.Image(src="", width=400, height=400, fit=ft.ImageFit.CONTAIN, visible=False, border_radius=20)
    rec_img_container = ft.Container(
        content=rec_img_control,
        on_click=lambda _: pick_rec_image_dialog.pick_files(allow_multiple=False)
    )
    rec_img_placeholder = ft.Container(
        content=ft.Column([
            ft.Icon(ft.Icons.FACE_RETOUCHING_NATURAL, size=60, color=ft.Colors.WHITE70),
            ft.Text("Click to upload a query photo", color=ft.Colors.WHITE70, size=16)
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=400, height=400,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLACK),
        border=ft.border.all(2, ft.Colors.with_opacity(0.3, ft.Colors.WHITE)),
        border_radius=20,
        alignment=ft.alignment.center,
        on_click=lambda _: pick_rec_image_dialog.pick_files(allow_multiple=False)
    )
    rec_result_view = ft.ListView(expand=True, spacing=10, padding=20)
    rec_img_path = None

    # Verification tab widgets
    ver_img1_control = ft.Image(src="", width=300, height=300, fit=ft.ImageFit.CONTAIN, visible=False, border_radius=20)
    ver_img1_container = ft.Container(
        content=ver_img1_control,
        on_click=lambda _: pick_ver1_image_dialog.pick_files(allow_multiple=False)
    )
    ver_img1_placeholder = ft.Container(
        content=ft.Column([ft.Icon(ft.Icons.IMAGE, size=40, color=ft.Colors.WHITE70), ft.Text("Photo 1", color=ft.Colors.WHITE70)], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=300, height=300,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLACK),
        border=ft.border.all(2, ft.Colors.with_opacity(0.3, ft.Colors.WHITE)),
        border_radius=20,
        alignment=ft.alignment.center,
        on_click=lambda _: pick_ver1_image_dialog.pick_files(allow_multiple=False)
    )

    ver_img2_control = ft.Image(src="", width=300, height=300, fit=ft.ImageFit.CONTAIN, visible=False, border_radius=20)
    ver_img2_container = ft.Container(
        content=ver_img2_control,
        on_click=lambda _: pick_ver2_image_dialog.pick_files(allow_multiple=False)
    )
    ver_img2_placeholder = ft.Container(
        content=ft.Column([ft.Icon(ft.Icons.IMAGE, size=40, color=ft.Colors.WHITE70), ft.Text("Photo 2", color=ft.Colors.WHITE70)], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=300, height=300,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLACK),
        border=ft.border.all(2, ft.Colors.with_opacity(0.3, ft.Colors.WHITE)),
        border_radius=20,
        alignment=ft.alignment.center,
        on_click=lambda _: pick_ver2_image_dialog.pick_files(allow_multiple=False)
    )
    ver_result_text = ft.Text("Please upload two photos to verify", size=16)
    ver_path1 = None
    ver_path2 = None

    # Quality analysis tab widgets
    qual_img_control = ft.Image(src="", width=400, height=400, fit=ft.ImageFit.CONTAIN, visible=False, border_radius=20)
    qual_img_container = ft.Container(
        content=qual_img_control,
        on_click=lambda _: pick_qual_image_dialog.pick_files(allow_multiple=False)
    )
    qual_img_placeholder = ft.Container(
        content=ft.Column([ft.Icon(ft.Icons.ANALYTICS, size=50, color=ft.Colors.WHITE70), ft.Text("Click to upload for analysis", color=ft.Colors.WHITE70)], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=400, height=400,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLACK),
        border=ft.border.all(2, ft.Colors.with_opacity(0.3, ft.Colors.WHITE)),
        border_radius=20,
        alignment=ft.alignment.center,
        on_click=lambda _: pick_qual_image_dialog.pick_files(allow_multiple=False)
    )
    qual_face_placeholder = ft.Container(
        width=150,
        height=150,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLACK),
        border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.WHITE)),
        border_radius=10,
        alignment=ft.alignment.center,
        content=ft.Text("No face", size=12, color=ft.Colors.WHITE70)
    )
    qual_face_img = ft.Image(width=150, height=150, fit=ft.ImageFit.CONTAIN, border_radius=10, visible=False)
    qual_result_text = ft.Text("", size=14)
    qual_img_path = None

    # --- 逻辑函数 ---

    def init_pipeline():
        nonlocal pipeline, quality_threshold
        status_text.value = f"Initializing {current_model_type} model..."
        page.update()
        default_path = db_paths.get(current_model_type) or get_db_path(current_model_type)
        set_db_path_value(default_path)
        try:
            device = 'cuda' if use_gpu else 'cpu'

            if current_model_type == "insightface":
                # InsightFace settings
                quality_threshold = 0.5
                qual_slider.min = 0.1
                qual_slider.max = 1.0
                qual_slider.divisions = 90
                qual_slider.value = quality_threshold
                qual_label.value = f"Quality threshold: {quality_threshold:.2f}"

                pipeline = InsightFacePipeline(
                    model_name='antelopev2',
                    device=device,
                    quality_threshold=quality_threshold,
                    similarity_threshold=similarity_threshold
                )
            else:
                # MagFace settings
                quality_threshold = 15.0  # Lowered from 20.0 to avoid "low quality" errors
                qual_slider.min = 10.0
                qual_slider.max = 60.0
                qual_slider.divisions = 50
                qual_slider.value = quality_threshold
                qual_label.value = f"Quality threshold: {quality_threshold:.2f}"

                model_path = 'pretrain/magface_epoch_00025.pth'
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"MagFace model not found at {model_path}")

                pipeline = FaceRecognitionPipeline(
                    model_path=model_path,
                    device=device,
                    quality_threshold=quality_threshold,
                    similarity_threshold=similarity_threshold
                )

            db_path = get_current_db_path()
            if os.path.exists(db_path):
                pipeline.load_database(db_path)
                status_text.value = f"{current_model_type} ready | Database: {len(pipeline.database['names'])} people"
            else:
                status_text.value = f"{current_model_type} ready | Database empty"

            page.update()
        except Exception as e:
            status_text.value = f"Initialization failed: {e}"
            page.open(ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(str(e))))
        page.update()

    def on_reg_file_picked(e: ft.FilePickerResultEvent):
        nonlocal reg_img_path
        if e.files:
            reg_img_path = e.files[0].path
            reg_img_control.src = reg_img_path
            reg_img_control.visible = True
            reg_img_placeholder.visible = False
            page.update()

    def do_register(e):
        if not pipeline: return
        name = reg_name_field.value
        if not name or not reg_img_path:
            page.open(ft.SnackBar(ft.Text("Please enter a name and select a photo")))
            return

        try:
            img = cv2.imdecode(np.fromfile(reg_img_path, dtype=np.uint8), -1)
            result = pipeline.register(name, img)

            if result['success']:
                page.open(ft.SnackBar(ft.Text(f"Registration succeeded! Quality: {result['quality']['magnitude']:.2f}"), bgcolor=ft.Colors.GREEN))
                reg_name_field.value = ""
                reg_img_control.visible = False
                reg_img_placeholder.visible = True
                status_text.value = f"{current_model_type} ready | Database: {len(pipeline.database['names'])} people"
            else:
                page.open(ft.AlertDialog(title=ft.Text("Registration failed"), content=ft.Text(result['message'])))
        except Exception as ex:
            page.open(ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(str(ex))))
        page.update()

    def on_rec_file_picked(e: ft.FilePickerResultEvent):
        nonlocal rec_img_path
        if e.files:
            rec_img_path = e.files[0].path
            rec_img_control.src = rec_img_path
            rec_img_control.visible = True
            rec_img_placeholder.visible = False
            rec_result_view.controls.clear()
            page.update()

    def do_recognize(e):
        if not pipeline or not rec_img_path: return
        try:
            img = cv2.imdecode(np.fromfile(rec_img_path, dtype=np.uint8), -1)
            result = pipeline.recognize(img, top_k=3)

            rec_result_view.controls.clear()

            if result['success']:
                if result['is_recognized']:
                    best = result['best_match']
                    rec_result_view.controls.append(
                        ft.Container(
                            content=ft.Column([
                                ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN, size=40),
                                ft.Text(f"Match found: {best['name']}", size=18, weight=ft.FontWeight.BOLD),
                                ft.Text(f"Similarity: {best['similarity']:.2%}", size=15, color=ft.Colors.GREEN)
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                            padding=20, bgcolor=ft.Colors.GREEN_50, border_radius=10
                        )
                    )
                else:
                    rec_result_view.controls.append(
                        ft.Container(
                            content=ft.Column([
                                ft.Icon(ft.Icons.WARNING, color=ft.Colors.ORANGE, size=40),
                                ft.Text("Unknown person", size=18, weight=ft.FontWeight.BOLD),
                                ft.Text("No enrolled identity matched", size=15)
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                            padding=20, bgcolor=ft.Colors.ORANGE_50, border_radius=10
                        )
                    )

                if result['all_results']:
                    rec_result_view.controls.append(ft.Text("Candidates:", size=15))
                    for r in result['all_results']:
                        icon = ft.Icons.STAR if r['is_match'] else ft.Icons.CIRCLE_OUTLINED
                        color = ft.Colors.BLUE if r['is_match'] else ft.Colors.GREY
                        rec_result_view.controls.append(
                            ft.ListTile(
                                leading=ft.Icon(icon, color=color),
                                title=ft.Text(r['name']),
                                subtitle=ft.Text(f"Similarity: {r['similarity']:.2%}")
                            )
                        )
            else:
                rec_result_view.controls.append(ft.Text(f"Error: {result['message']}", color=ft.Colors.RED))
        except Exception as ex:
            page.open(ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(str(ex))))
        page.update()

    def on_ver1_picked(e):
        nonlocal ver_path1
        if e.files:
            ver_path1 = e.files[0].path
            ver_img1_control.src = ver_path1
            ver_img1_control.visible = True
            ver_img1_placeholder.visible = False
            page.update()

    def on_ver2_picked(e):
        nonlocal ver_path2
        if e.files:
            ver_path2 = e.files[0].path
            ver_img2_control.src = ver_path2
            ver_img2_control.visible = True
            ver_img2_placeholder.visible = False
            page.update()

    def do_verify(e):
        if not pipeline or not ver_path1 or not ver_path2: return
        try:
            img1 = cv2.imdecode(np.fromfile(ver_path1, dtype=np.uint8), -1)
            img2 = cv2.imdecode(np.fromfile(ver_path2, dtype=np.uint8), -1)

            result = pipeline.verify(img1, img2)

            if result['success']:
                sim = result['similarity']
                is_same = result['is_same_person']
                ver_result_text.value = f"{'✅ Same person' if is_same else '❌ Different people'} ({sim:.2%})"
                ver_result_text.color = ft.Colors.GREEN if is_same else ft.Colors.RED
            else:
                ver_result_text.value = f"Verification failed: {result['message']}"
                ver_result_text.color = ft.Colors.GREY
        except Exception as ex:
            page.open(ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(str(ex))))
        page.update()

    def on_qual_picked(e):
        nonlocal qual_img_path
        if e.files:
            qual_img_path = e.files[0].path
            qual_img_control.src = qual_img_path
            qual_img_control.visible = True
            qual_img_placeholder.visible = False
            page.update()

    def do_quality(e):
        if not pipeline or not qual_img_path:
            return
        try:
            img = cv2.imdecode(np.fromfile(qual_img_path, dtype=np.uint8), -1)
            aligned_face = None
            detection_confidence = None
            quality = None

            if current_model_type == "insightface":
                # InsightFace: extract_features(img) -> (embedding, det_score, aligned_face)
                embedding, det_score, aligned_face = pipeline.extract_features(img)
                if embedding is None:
                    qual_result_text.value = "No face detected"
                    qual_face_img.visible = False
                    qual_face_img.src = None
                    qual_face_img.src_base64 = None
                    page.update()
                    return
                quality = pipeline.assess_quality(det_score)
                detection_confidence = det_score
            else:
                # MagFace: 先 detect_and_align(img) -> aligned_face, 再 extract_features(aligned_face) -> (embedding, magnitude)
                aligned_face = pipeline.detect_and_align(img)
                if aligned_face is None:
                    qual_result_text.value = "No face detected"
                    qual_face_img.visible = False
                    qual_face_img.src = None
                    qual_face_img.src_base64 = None
                    page.update()
                    return
                _, magnitude = pipeline.extract_features(aligned_face)
                quality = pipeline.assess_quality(magnitude)

            # 显示对齐后的人脸
            if aligned_face is not None:
                b64 = cv2_to_base64(aligned_face)
                qual_face_img.src_base64 = b64
                qual_face_img.visible = True
                qual_face_img.update()
            else:
                qual_face_img.src = None
                qual_face_img.src_base64 = None
                qual_face_img.visible = False

            # 构建结果显示
            det_text = f"{detection_confidence:.4f}" if detection_confidence is not None else "N/A (MagFace uses magnitude)"
            qual_result_text.value = (
                f"Quality score: {quality['magnitude']:.2f}\n"
                f"Level: {quality['level']}\n"
                f"Description: {quality['description']}\n"
                f"Detection confidence: {det_text}\n"
                f"Acceptable: {'Yes' if quality['is_acceptable'] else 'No'}"
            )
        except Exception as ex:
            page.open(ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(str(ex))))
        page.update()

    def browse_db_path(e):
        if db_path_picker:
            db_path_picker.pick_files(allow_multiple=False)

    def save_db(e):
        if pipeline:
            try:
                db_path = get_current_db_path()
                dir_name = os.path.dirname(db_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                pipeline.save_database(db_path)
                page.open(ft.SnackBar(ft.Text(f"Database saved to {db_path}")))
            except Exception as ex:
                page.open(ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(str(ex))))

    def load_db(e):
        if pipeline:
            try:
                db_path = get_current_db_path()
                pipeline.load_database(db_path)
                status_text.value = f"{current_model_type} ready | Database: {len(pipeline.database['names'])} people"
                page.open(ft.SnackBar(ft.Text(f"Database loaded from {db_path}")))
                page.update()
            except Exception as ex:
                page.open(ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(str(ex))))

    def delete_db(e):
        if not pipeline:
            page.open(ft.SnackBar(ft.Text("Pipeline not initialized")))
            return

        db_path = get_current_db_path()
        entry_count = len(pipeline.database.get('names', []))

        def close_dialog(_=None):
            confirm_dialog.open = False
            page.update()

        def confirm_delete(_):
            try:
                result = pipeline.clear_database(db_path)
                status_text.value = f"{current_model_type} ready | Database empty"
                msg = result.get('message', 'Database cleared')
                if result.get('removed_file'):
                    msg += " (file removed)"
                page.open(ft.SnackBar(ft.Text(msg)))
            except Exception as ex:
                page.open(ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(str(ex))))
            finally:
                close_dialog()

        confirm_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Delete database?"),
            content=ft.Text(
                f"This will remove {entry_count} entr{'y' if entry_count==1 else 'ies'} from memory and delete the file if it exists.\nPath: {db_path}\nThis action cannot be undone."
            ),
            actions=[
                ft.TextButton("Cancel", on_click=close_dialog),
                ft.TextButton(
                    "Delete",
                    style=ft.ButtonStyle(color=ft.Colors.RED_600),
                    on_click=confirm_delete
                )
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )

        page.open(confirm_dialog)

    def show_registered_list(e):
        if not pipeline: return

        def delete_user(name):
            if pipeline.remove(name):
                dlg_modal.open = False
                page.update()
                show_registered_list(None) # Refresh
                status_text.value = f"{current_model_type} ready | Database: {len(pipeline.database['names'])} people"
                page.update()

        names = pipeline.list_registered()

        lv = ft.ListView(expand=True, spacing=10)
        for name in names:
            lv.controls.append(
                ft.ListTile(
                    title=ft.Text(name),
                    trailing=ft.IconButton(ft.Icons.DELETE, icon_color=ft.Colors.RED,
                                         on_click=lambda _, n=name: delete_user(n))
                )
            )

        dlg_modal = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Registered users ({len(names)})", size=18),
            content=ft.Container(content=lv, width=400, height=500),
            actions=[ft.TextButton("Close", on_click=lambda _: page.close(dlg_modal))],
        )
        page.open(dlg_modal)

    def update_settings(e):
        nonlocal use_gpu, quality_threshold, similarity_threshold
        use_gpu = gpu_switch.value
        quality_threshold = qual_slider.value
        similarity_threshold = sim_slider.value

        qual_label.value = f"Quality threshold: {quality_threshold:.2f}"
        sim_label.value = f"Similarity threshold: {similarity_threshold:.2f}"

        if pipeline:
            pipeline.quality_threshold = quality_threshold
            pipeline.similarity_threshold = similarity_threshold
        page.update()

    def reload_model(e):
        update_settings(None)
        init_pipeline()

    # --- File Pickers ---
    pick_reg_image_dialog = ft.FilePicker(on_result=on_reg_file_picked)
    pick_rec_image_dialog = ft.FilePicker(on_result=on_rec_file_picked)
    pick_ver1_image_dialog = ft.FilePicker(on_result=on_ver1_picked)
    pick_ver2_image_dialog = ft.FilePicker(on_result=on_ver2_picked)
    pick_qual_image_dialog = ft.FilePicker(on_result=on_qual_picked)
    db_path_picker = ft.FilePicker(on_result=on_db_path_selected)

    page.overlay.extend([
        pick_reg_image_dialog, pick_rec_image_dialog,
        pick_ver1_image_dialog, pick_ver2_image_dialog,
        pick_qual_image_dialog, db_path_picker
    ])

    # --- Sidebar ---
    gpu_switch = ft.Switch(label="Use GPU acceleration", value=True, on_change=reload_model, active_color=ft.Colors.BLUE_600)
    qual_slider = ft.Slider(min=0.1, max=1.0, divisions=90, value=0.5, label="{value}", on_change=update_settings, active_color=ft.Colors.BLUE_600)
    qual_label = ft.Text("Quality threshold: 0.50", size=12, color=ft.Colors.GREY_700)
    sim_slider = ft.Slider(min=0.2, max=0.8, divisions=60, value=0.4, label="{value}", on_change=update_settings, active_color=ft.Colors.BLUE_600)
    sim_label = ft.Text("Similarity threshold: 0.40", size=12, color=ft.Colors.GREY_700)

    def style_button(btn, has_border=False):
        side = ft.BorderSide(1, ft.Colors.BLUE_200) if has_border else None
        btn.style = ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            overlay_color=ft.Colors.BLUE_50,
            side=side
        )
        return btn

    # --- Theme Management ---
    bg_gradient_light = ft.LinearGradient(
        begin=ft.alignment.top_left,
        end=ft.alignment.bottom_right,
        colors=["#2980b9", "#6dd5fa", "#b3e5fc"],
    )
    bg_gradient_dark = ft.LinearGradient(
        begin=ft.alignment.top_left,
        end=ft.alignment.bottom_right,
        colors=["#0f2027", "#203a43", "#2c5364"],
    )

    bg_container = ft.Container(
        expand=True,
        gradient=bg_gradient_light
    )

    sidebar_title = ft.Text("Face Recognition", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_900)
    sidebar_sys_label = ft.Text("System settings", size=14, weight=ft.FontWeight.W_500, color=ft.Colors.GREY_600)
    sidebar_db_label = ft.Text("Database management", size=14, weight=ft.FontWeight.W_500, color=ft.Colors.GREY_600)
    db_path_hint = ft.Text("Select where to save/load .pkl files", size=11, color=ft.Colors.GREY_500)
    db_path_controls = ft.Row(
        [
            db_path_field,
            ft.IconButton(ft.Icons.FOLDER_OPEN, icon_color=ft.Colors.BLUE_700, tooltip="Browse", on_click=browse_db_path)
        ],
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.END,
        alignment=ft.MainAxisAlignment.START
    )
    sidebar_theme_label = ft.Text("Switch theme", size=12, color=ft.Colors.GREY_600)

    def change_theme(mode):
        page.theme_mode = mode
        if mode == ft.ThemeMode.LIGHT:
            bg_container.gradient = bg_gradient_light
            sidebar.bgcolor = ft.Colors.with_opacity(0.95, "#f0f4f8")
            sidebar_title.color = ft.Colors.BLUE_900
            sidebar_sys_label.color = ft.Colors.GREY_600
            sidebar_db_label.color = ft.Colors.GREY_600
            sidebar_theme_label.color = ft.Colors.GREY_600
            db_path_hint.color = ft.Colors.GREY_500
            qual_label.color = ft.Colors.GREY_700
            sim_label.color = ft.Colors.GREY_700
            status_frame.bgcolor = ft.Colors.with_opacity(0.5, ft.Colors.WHITE)
            status_text.color = ft.Colors.BLACK
        else:
            bg_container.gradient = bg_gradient_dark
            sidebar.bgcolor = ft.Colors.with_opacity(0.95, "#1a1a1a")
            sidebar_title.color = ft.Colors.BLUE_100
            sidebar_sys_label.color = ft.Colors.GREY_400
            sidebar_db_label.color = ft.Colors.GREY_400
            sidebar_theme_label.color = ft.Colors.GREY_400
            db_path_hint.color = ft.Colors.GREY_400
            qual_label.color = ft.Colors.GREY_400
            sim_label.color = ft.Colors.GREY_400
            status_frame.bgcolor = ft.Colors.with_opacity(0.5, ft.Colors.BLACK)
            status_text.color = ft.Colors.WHITE70
        page.update()

    sidebar = ft.Container(
        width=300,
        bgcolor=ft.Colors.with_opacity(0.95, "#f0f4f8"),
        padding=30,
        border_radius=ft.border_radius.only(top_right=20, bottom_right=20),
        shadow=ft.BoxShadow(blur_radius=10, color=ft.Colors.with_opacity(0.1, ft.Colors.BLACK)),
        content=ft.Column([
            sidebar_title,
            ft.Container(height=20),
            sidebar_sys_label,
            gpu_switch,
            ft.Container(height=10),
            qual_label, qual_slider,
            sim_label, sim_slider,
            ft.Divider(color=ft.Colors.GREY_300),
            sidebar_db_label,
            db_path_hint,
            db_path_controls,
            style_button(ft.ElevatedButton("Save database", icon=ft.Icons.SAVE, on_click=save_db, width=240, bgcolor=ft.Colors.WHITE, color=ft.Colors.BLUE_700, elevation=0), has_border=True),
            style_button(ft.ElevatedButton("Load database", icon=ft.Icons.UPLOAD_FILE, on_click=load_db, width=240, bgcolor=ft.Colors.WHITE, color=ft.Colors.BLUE_700, elevation=0), has_border=True),
            style_button(ft.ElevatedButton("Delete database", icon=ft.Icons.DELETE, on_click=delete_db, width=240, bgcolor=ft.Colors.RED_50, color=ft.Colors.RED_700, elevation=0), has_border=True),
            style_button(ft.OutlinedButton("View registered list", icon=ft.Icons.LIST, on_click=show_registered_list, width=240, style=ft.ButtonStyle(color=ft.Colors.BLUE_700, side=ft.BorderSide(1, ft.Colors.BLUE_200)))),
            ft.Container(expand=True),
            ft.Row([
                ft.IconButton(ft.Icons.SUNNY, icon_color=ft.Colors.ORANGE, on_click=lambda _: change_theme(ft.ThemeMode.LIGHT)),
                ft.IconButton(ft.Icons.NIGHTLIGHT, icon_color=ft.Colors.BLUE_GREY, on_click=lambda _: change_theme(ft.ThemeMode.DARK)),
                sidebar_theme_label
            ])
        ])
    )

    # --- Main content area ---
    def glass_container(content, padding=40):
        return ft.Container(
            content=content,
            padding=padding,
            bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.WHITE),
            border=ft.border.all(1, ft.Colors.with_opacity(0.2, ft.Colors.WHITE)),
            border_radius=20,
            shadow=ft.BoxShadow(blur_radius=15, color=ft.Colors.with_opacity(0.1, ft.Colors.BLACK)),
            alignment=ft.alignment.center,
            expand=True
        )

    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        indicator_color=ft.Colors.WHITE,
        label_color=ft.Colors.WHITE,
        unselected_label_color=ft.Colors.WHITE54,
        divider_color=ft.Colors.TRANSPARENT,
        tabs=[
            ft.Tab(
                tab_content=ft.Row([
                    ft.Icon(ft.Icons.PERSON_ADD, size=18),
                    ft.Text("Register", size=14, weight=ft.FontWeight.W_500)
                ], spacing=5, alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                content=glass_container(
                    ft.Column([
                        ft.Text("Register new user", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                        ft.Container(height=20),
                        ft.Row([
                            ft.Column([
                                ft.Stack([reg_img_placeholder, reg_img_container]),
                                ft.Text("Click the area above to upload", size=12, color=ft.Colors.WHITE70)
                            ]),
                            ft.Container(width=40),
                            ft.Column([
                                reg_name_field,
                                ft.Container(height=20),
                                ft.Container(
                                    content=ft.ElevatedButton(
                                        "Confirm registration",
                                        icon=ft.Icons.CHECK,
                                        color=ft.Colors.WHITE,
                                        bgcolor=ft.Colors.TRANSPARENT,
                                        width=200, height=50,
                                        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=25), elevation=0),
                                        on_click=do_register
                                    ),
                                    gradient=ft.LinearGradient(colors=[ft.Colors.BLUE_400, ft.Colors.BLUE_700]),
                                    border_radius=25,
                                    shadow=ft.BoxShadow(blur_radius=10, color=ft.Colors.BLUE_200)
                                )
                            ], alignment=ft.MainAxisAlignment.CENTER)
                        ], alignment=ft.MainAxisAlignment.CENTER)
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                )
            ),
            ft.Tab(
                tab_content=ft.Row([
                    ft.Icon(ft.Icons.FACE, size=18),
                    ft.Text("Recognize", size=14, weight=ft.FontWeight.W_500)
                ], spacing=5, alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                content=glass_container(
                    ft.Row([
                        ft.Container(
                            expand=1,
                            content=ft.Column([
                                ft.Text("Query photo", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                                ft.Stack([rec_img_placeholder, rec_img_container]),
                                ft.Container(height=20),
                                ft.Container(
                                    content=ft.ElevatedButton(
                                        "Start recognition",
                                        icon=ft.Icons.SEARCH,
                                        color=ft.Colors.WHITE,
                                        bgcolor=ft.Colors.TRANSPARENT,
                                        width=200, height=50,
                                        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=25), elevation=0),
                                        on_click=do_recognize
                                    ),
                                    gradient=ft.LinearGradient(colors=[ft.Colors.BLUE_400, ft.Colors.BLUE_700]),
                                    border_radius=25
                                )
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                        ),
                        ft.VerticalDivider(color=ft.Colors.WHITE24),
                        ft.Container(
                            expand=1,
                            content=ft.Column([
                                ft.Text("Recognition results", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                                ft.Container(
                                    content=rec_result_view,
                                    bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.WHITE),
                                    border_radius=10,
                                    expand=True,
                                    padding=10
                                )
                            ])
                        )
                    ])
                )
            ),
            ft.Tab(
                tab_content=ft.Row([
                    ft.Icon(ft.Icons.COMPARE, size=18),
                    ft.Text("1:1 Verify", size=14, weight=ft.FontWeight.W_500)
                ], spacing=5, alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                content=glass_container(
                    ft.Column([
                        ft.Container(
                            padding=20,
                            bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.WHITE),
                            border_radius=10,
                            content=ft.Column([
                                ver_result_text
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                        ),
                        ft.Container(height=20),
                        ft.Row([
                            ft.Column([
                                ft.Stack([ver_img1_placeholder, ver_img1_container]),
                                ft.Text("Click above to upload Photo 1", size=12, color=ft.Colors.WHITE70)
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.Container(width=50),
                            ft.Column([
                                ft.Stack([ver_img2_placeholder, ver_img2_container]),
                                ft.Text("Click above to upload Photo 2", size=12, color=ft.Colors.WHITE70)
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                        ], alignment=ft.MainAxisAlignment.CENTER),
                        ft.Container(height=40),
                        ft.Container(
                            content=ft.ElevatedButton(
                                "Start verification",
                                icon=ft.Icons.COMPARE_ARROWS,
                                color=ft.Colors.WHITE,
                                bgcolor=ft.Colors.TRANSPARENT,
                                width=300, height=50,
                                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=25), elevation=0),
                                on_click=do_verify
                            ),
                            gradient=ft.LinearGradient(colors=[ft.Colors.BLUE_400, ft.Colors.BLUE_700]),
                            border_radius=25
                        )
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                )
            ),
            ft.Tab(
                tab_content=ft.Row([
                    ft.Icon(ft.Icons.ANALYTICS, size=18),
                    ft.Text("Quality analysis", size=14, weight=ft.FontWeight.W_500)
                ], spacing=5, alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                content=glass_container(
                    ft.Row([
                        ft.Container(
                            expand=1,
                            content=ft.Column([
                                ft.Text("Original photo", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                                ft.Stack([qual_img_placeholder, qual_img_container]),
                                ft.Container(height=20),
                                ft.Container(
                                    content=ft.ElevatedButton(
                                        "Analyze quality",
                                        icon=ft.Icons.ANALYTICS,
                                        color=ft.Colors.WHITE,
                                        bgcolor=ft.Colors.TRANSPARENT,
                                        width=200, height=50,
                                        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=25), elevation=0),
                                        on_click=do_quality
                                    ),
                                    gradient=ft.LinearGradient(colors=[ft.Colors.BLUE_400, ft.Colors.BLUE_700]),
                                    border_radius=25
                                )
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                        ),
                        ft.VerticalDivider(color=ft.Colors.WHITE24),
                        ft.Container(
                            expand=1,
                            content=ft.Column([
                                ft.Text("Analysis report", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                                ft.Row([
                                    ft.Text("Detected face:", color=ft.Colors.WHITE),
                                    ft.Stack([qual_face_placeholder, qual_face_img])
                                ]),
                                ft.Container(
                                    padding=20,
                                    bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.WHITE),
                                    border_radius=10,
                                    content=qual_result_text
                                )
                            ])
                        )
                    ])
                )
            )
        ],
        expand=True,
    )

    # --- Page layout ---
    status_frame = ft.Container(
        content=status_text,
        padding=5,
        bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.WHITE),
        border_radius=ft.border_radius.only(top_left=10, top_right=10)
    )

    page.add(
        ft.Stack(
            [
                # Background Gradient
                bg_container,
                # Main Content
                ft.Row(
                    [
                        sidebar,
                        ft.VerticalDivider(width=1, color=ft.Colors.TRANSPARENT),
                        ft.Column([
                            ft.Container(height=10), # Top spacing
                            tabs,
                            status_frame
                        ], expand=True)
                    ],
                    expand=True,
                )
            ],
            expand=True
        )
    )

    # Start initialization thread
    threading.Thread(target=init_pipeline, daemon=True).start()

if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.FLET_APP)
