import flet as ft
import cv2
import numpy as np
import base64
import os
import sys
import threading
from typing import Optional

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.insightface_pipeline import InsightFacePipeline

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
    pipeline: Optional[InsightFacePipeline] = None
    db_path = 'database/face_database.pkl'

    # State variables
    use_gpu = True
    quality_threshold = 0.5
    similarity_threshold = 0.4

    # --- UI component references ---
    status_text = ft.Text("Ready", size=12)

    # Registration tab widgets
    reg_name_field = ft.TextField(label="Name", width=300)
    reg_img_control = ft.Image(src="", width=400, height=400, fit=ft.ImageFit.CONTAIN, visible=False)
    reg_img_placeholder = ft.Container(
        content=ft.Column([
            ft.Icon(ft.Icons.ADD_A_PHOTO, size=50, color=ft.Colors.GREY_400),
            ft.Text("Click to upload", color=ft.Colors.GREY_600)
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=400, height=400, bgcolor=ft.Colors.GREY_100, border_radius=10,
        alignment=ft.alignment.center,
        on_click=lambda _: pick_reg_image_dialog.pick_files(allow_multiple=False)
    )
    reg_img_path = None

    # Recognition tab widgets
    rec_img_control = ft.Image(src="", width=400, height=400, fit=ft.ImageFit.CONTAIN, visible=False)
    rec_img_placeholder = ft.Container(
        content=ft.Column([
            ft.Icon(ft.Icons.FACE_RETOUCHING_NATURAL, size=50, color=ft.Colors.GREY_400),
            ft.Text("Click to upload a query photo", color=ft.Colors.GREY_600)
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=400, height=400, bgcolor=ft.Colors.GREY_100, border_radius=10,
        alignment=ft.alignment.center,
        on_click=lambda _: pick_rec_image_dialog.pick_files(allow_multiple=False)
    )
    rec_result_view = ft.ListView(expand=True, spacing=10, padding=20)
    rec_img_path = None

    # Verification tab widgets
    ver_img1_control = ft.Image(src="", width=300, height=300, fit=ft.ImageFit.CONTAIN, visible=False)
    ver_img1_placeholder = ft.Container(
        content=ft.Column([ft.Icon(ft.Icons.IMAGE, size=40), ft.Text("Photo 1")], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=300, height=300, bgcolor=ft.Colors.GREY_100, border_radius=10, alignment=ft.alignment.center,
        on_click=lambda _: pick_ver1_image_dialog.pick_files(allow_multiple=False)
    )

    ver_img2_control = ft.Image(src="", width=300, height=300, fit=ft.ImageFit.CONTAIN, visible=False)
    ver_img2_placeholder = ft.Container(
        content=ft.Column([ft.Icon(ft.Icons.IMAGE, size=40), ft.Text("Photo 2")], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=300, height=300, bgcolor=ft.Colors.GREY_100, border_radius=10, alignment=ft.alignment.center,
        on_click=lambda _: pick_ver2_image_dialog.pick_files(allow_multiple=False)
    )
    ver_result_text = ft.Text("Please upload two photos to verify", size=16)
    ver_path1 = None
    ver_path2 = None

    # Quality analysis tab widgets
    qual_img_control = ft.Image(src="", width=400, height=400, fit=ft.ImageFit.CONTAIN, visible=False)
    qual_img_placeholder = ft.Container(
        content=ft.Column([ft.Icon(ft.Icons.ANALYTICS, size=50), ft.Text("Click to upload for analysis")], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=400, height=400, bgcolor=ft.Colors.GREY_100, border_radius=10, alignment=ft.alignment.center,
        on_click=lambda _: pick_qual_image_dialog.pick_files(allow_multiple=False)
    )
    qual_face_placeholder = ft.Container(
        width=150,
        height=150,
        bgcolor=ft.Colors.GREY_100,
        border_radius=10,
        alignment=ft.alignment.center,
        content=ft.Text("No face", size=12, color=ft.Colors.GREY_600)
    )
    qual_face_img = ft.Image(width=150, height=150, fit=ft.ImageFit.CONTAIN, border_radius=10, visible=False)
    qual_result_text = ft.Text("", size=14)
    qual_img_path = None

    # --- 逻辑函数 ---

    def init_pipeline():
        nonlocal pipeline
        status_text.value = "Initializing model..."
        page.update()
        try:
            device = 'cuda' if use_gpu else 'cpu'
            pipeline = InsightFacePipeline(
                model_name='buffalo_l',
                device=device,
                quality_threshold=quality_threshold,
                similarity_threshold=similarity_threshold
            )
            if os.path.exists(db_path):
                pipeline.load_database(db_path)
                status_text.value = f"Model ready | Database: {len(pipeline.database['names'])} people"
            else:
                status_text.value = "Model ready | Database empty"
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
                status_text.value = f"Model ready | Database: {len(pipeline.database['names'])} people"
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
        if not pipeline or not qual_img_path: return
        try:
            img = cv2.imdecode(np.fromfile(qual_img_path, dtype=np.uint8), -1)
            embedding, det_score, aligned_face = pipeline.extract_features(img)

            if embedding is not None:
                quality = pipeline.assess_quality(det_score)

                if aligned_face is not None:
                    b64 = cv2_to_base64(aligned_face)
                    qual_face_img.src_base64 = b64
                    qual_face_img.visible = True
                    qual_face_img.update()
                else:
                    qual_face_img.src = None
                    qual_face_img.src_base64 = None
                    qual_face_img.visible = False

                qual_result_text.value = (
                    f"Quality score: {quality['magnitude']:.2f}\n"
                    f"Level: {quality['level']}\n"
                    f"Description: {quality['description']}\n"
                    f"Detection confidence: {det_score:.4f}\n"
                    f"Acceptable: {'Yes' if quality['is_acceptable'] else 'No'}"
                )
            else:
                qual_result_text.value = "No face detected"
                qual_face_img.src = None
                qual_face_img.src_base64 = None
                qual_face_img.visible = False
                qual_face_img.update()
        except Exception as ex:
            page.open(ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(str(ex))))
        page.update()

    def save_db(e):
        if pipeline:
            try:
                os.makedirs('database', exist_ok=True)
                pipeline.save_database(db_path)
                page.open(ft.SnackBar(ft.Text("Database saved successfully")))
            except Exception as ex:
                page.open(ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(str(ex))))

    def load_db(e):
        if pipeline:
            try:
                pipeline.load_database(db_path)
                status_text.value = f"Model ready | Database: {len(pipeline.database['names'])} people"
                page.open(ft.SnackBar(ft.Text("Database loaded successfully")))
                page.update()
            except Exception as ex:
                page.open(ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(str(ex))))

    def show_registered_list(e):
        if not pipeline: return

        def delete_user(name):
            if pipeline.remove(name):
                dlg_modal.open = False
                page.update()
                show_registered_list(None) # Refresh
                status_text.value = f"Model ready | Database: {len(pipeline.database['names'])} people"
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

    page.overlay.extend([
        pick_reg_image_dialog, pick_rec_image_dialog,
        pick_ver1_image_dialog, pick_ver2_image_dialog,
        pick_qual_image_dialog
    ])

    # --- Sidebar ---
    gpu_switch = ft.Switch(label="Use GPU acceleration", value=True, on_change=reload_model)
    qual_slider = ft.Slider(min=0.1, max=1.0, divisions=90, value=0.5, label="{value}", on_change=update_settings)
    qual_label = ft.Text("Quality threshold: 0.50")
    sim_slider = ft.Slider(min=0.2, max=0.8, divisions=60, value=0.4, label="{value}", on_change=update_settings)
    sim_label = ft.Text("Similarity threshold: 0.40")

    sidebar = ft.Container(
        width=280,
    bgcolor=ft.Colors.GREY_100,  # Light gray sidebar fits the white theme
        padding=20,
        content=ft.Column([
            ft.Text("Face Recognition", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_700),
            ft.Divider(),
            ft.Text("System settings", size=14),
            gpu_switch,
            ft.Divider(),
            qual_label, qual_slider,
            sim_label, sim_slider,
            ft.Divider(),
            ft.Text("Database management", size=14),
            ft.ElevatedButton("Save database", icon=ft.Icons.SAVE, on_click=save_db, width=240),
            ft.ElevatedButton("Load database", icon=ft.Icons.UPLOAD_FILE, on_click=load_db, width=240),
            ft.OutlinedButton("View registered list", icon=ft.Icons.LIST, on_click=show_registered_list, width=240),
            ft.Divider(),
            ft.Row([
                ft.IconButton(ft.Icons.SUNNY, on_click=lambda _: setattr(page, 'theme_mode', ft.ThemeMode.LIGHT) or page.update()),
                ft.IconButton(ft.Icons.NIGHTLIGHT, on_click=lambda _: setattr(page, 'theme_mode', ft.ThemeMode.DARK) or page.update()),
                ft.Text("Switch theme", size=13)
            ])
        ])
    )

    # --- Main content area ---
    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
                ft.Tab(
                    tab_content=ft.Row([
                        ft.Icon(ft.Icons.PERSON_ADD, size=18),
                        ft.Text("Register", size=14, weight=ft.FontWeight.W_400)
                    ], spacing=5, alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                content=ft.Container(
                    padding=40,
                    content=ft.Column([
                        ft.Text("Register new user", size=24, weight=ft.FontWeight.BOLD),
                        ft.Row([
                            ft.Column([
                                reg_img_placeholder,
                                reg_img_control,
                                ft.Text("Click the area above to upload", size=12, color=ft.Colors.GREY_600)
                            ]),
                            ft.Container(width=40),
                            ft.Column([
                                reg_name_field,
                                ft.ElevatedButton("Confirm registration", icon=ft.Icons.CHECK, bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE, width=200, height=50, on_click=do_register)
                            ], alignment=ft.MainAxisAlignment.CENTER)
                        ], alignment=ft.MainAxisAlignment.CENTER)
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                )
            ),
                ft.Tab(
                    tab_content=ft.Row([
                        ft.Icon(ft.Icons.FACE, size=18),
                        ft.Text("Recognize", size=14, weight=ft.FontWeight.W_400)
                    ], spacing=5, alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                content=ft.Container(
                    padding=20,
                    content=ft.Row([
                        ft.Container(
                            expand=1,
                            content=ft.Column([
                                ft.Text("Query photo", size=18, weight=ft.FontWeight.BOLD),
                                ft.Stack([rec_img_placeholder, rec_img_control]),
                                ft.ElevatedButton("Start recognition", icon=ft.Icons.SEARCH, width=200, height=50, on_click=do_recognize)
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                        ),
                        ft.VerticalDivider(),
                        ft.Container(
                            expand=1,
                            content=ft.Column([
                                ft.Text("Recognition results", size=18, weight=ft.FontWeight.BOLD),
                                rec_result_view
                            ])
                        )
                    ])
                )
            ),
                ft.Tab(
                    tab_content=ft.Row([
                        ft.Icon(ft.Icons.COMPARE, size=18),
                        ft.Text("1:1 Verify", size=14, weight=ft.FontWeight.W_400)
                    ], spacing=5, alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                content=ft.Container(
                    padding=20,
                    content=ft.Column([
                        ft.Container(
                            padding=20,
                            bgcolor=ft.Colors.GREY_100,
                            border_radius=10,
                            content=ft.Column([
                                ver_result_text
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                        ),
                        ft.Container(height=20),
                        ft.Row([
                            ft.Column([
                                ft.Stack([ver_img1_placeholder, ver_img1_control]),
                                ft.Text("Click above to upload Photo 1", size=12, color=ft.Colors.GREY_600)
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.Container(width=50),
                            ft.Column([
                                ft.Stack([ver_img2_placeholder, ver_img2_control]),
                                ft.Text("Click above to upload Photo 2", size=12, color=ft.Colors.GREY_600)
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                        ], alignment=ft.MainAxisAlignment.CENTER),
                        ft.Container(height=40),
                        ft.ElevatedButton("Start verification", icon=ft.Icons.COMPARE_ARROWS, width=300, height=50, on_click=do_verify)
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                )
            ),
                ft.Tab(
                    tab_content=ft.Row([
                        ft.Icon(ft.Icons.ANALYTICS, size=18),
                        ft.Text("Quality analysis", size=14, weight=ft.FontWeight.W_400)
                    ], spacing=5, alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                content=ft.Container(
                    padding=20,
                    content=ft.Row([
                        ft.Container(
                            expand=1,
                            content=ft.Column([
                                ft.Text("Original photo", size=18, weight=ft.FontWeight.BOLD),
                                ft.Stack([qual_img_placeholder, qual_img_control]),
                                ft.ElevatedButton("Analyze quality", icon=ft.Icons.ANALYTICS, width=200, height=50, on_click=do_quality)
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                        ),
                        ft.VerticalDivider(),
                        ft.Container(
                            expand=1,
                            content=ft.Column([
                                ft.Text("Analysis report", size=18, weight=ft.FontWeight.BOLD),
                                ft.Row([
                                    ft.Text("Detected face:"),
                                    ft.Stack([qual_face_placeholder, qual_face_img])
                                ]),
                                ft.Container(
                                    padding=20,
                                    bgcolor=ft.Colors.GREY_100,
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
    page.add(
        ft.Row(
            [
                sidebar,
                ft.VerticalDivider(width=1),
                ft.Column([
                    tabs,
                    status_frame := ft.Container(
                        content=status_text,
                        padding=5,
                        bgcolor=ft.Colors.GREY_100
                    )
                ], expand=True)
            ],
            expand=True,
        )
    )

    # Start initialization thread
    threading.Thread(target=init_pipeline, daemon=True).start()

if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.FLET_APP)
