# 
# Este script procesa imágenes de páginas de periódicos históricos por medio de un modelo YOLO para detectar y extraer columnas de texto.
# 
# El programa realiza las siguientes tareas principales:
# 1. Detecta encabezados, secciones y columnas en las imágenes.
# 2. Ajusta los límites de las columnas basándose en las secciones y encabezados detectados.
# 3. Elimina columnas superpuestas.
# 4. Ordena las columnas por sección.
# 5. Recorta y guarda las columnas como imágenes separadas.
# 
# Requisitos:
# - ultralytics
# - opencv-python (cv2)
# - numpy
# 
# El script espera un modelo YOLO entrenado en la ruta "./modelo/yolo11x-layout.pt".
# 

# Importaciones necesarias
from ultralytics import YOLO         # Para cargar y usar el modelo YOLO
import cv2                           # Para procesamiento de imágenes
import numpy as np                   # Para operaciones numéricas eficientes
from collections import defaultdict  # Para crear diccionarios con valores por defecto
import os                            # Para operaciones de sistema de archivos

def get_header_boundaries(boxes, classes, names):
    """
    Extrae las coordenadas de los encabezados detectados.

    Args:
    boxes (np.array): Array de cajas delimitadoras.
    classes (np.array): Array de clases correspondientes a cada caja.
    names (dict): Diccionario que mapea índices de clase a nombres de clase.

    Returns:
    list: Lista de coordenadas [x1, y1, x2, y2] de los encabezados detectados.
    """
    header_boxes = []
    for box, cls in zip(boxes, classes):
        class_name = names[int(cls)]
        if class_name == 'encabezado':
            x1, y1, x2, y2 = map(int, box.tolist())
            header_boxes.append([x1, y1, x2, y2])
    return header_boxes

def get_section_boundaries(boxes, classes, names):
    """
    Extrae las coordenadas de las secciones detectadas.

    Args:
    boxes (np.array): Array de cajas delimitadoras.
    classes (np.array): Array de clases correspondientes a cada caja.
    names (dict): Diccionario que mapea índices de clase a nombres de clase.

    Returns:
    list: Lista de coordenadas [x1, y1, x2, y2] de las secciones detectadas.
    """
    section_boxes = []
    for box, cls in zip(boxes, classes):
        class_name = names[int(cls)]
        if class_name == 'seccion':
            x1, y1, x2, y2 = map(int, box.tolist())
            section_boxes.append([x1, y1, x2, y2])
    return section_boxes

def get_containing_section(column_box, section_boxes):
    """
    Encuentra la sección que contiene una columna dada.

    Args:
    column_box (list): Coordenadas [x1, y1, x2, y2] de la columna.
    section_boxes (list): Lista de coordenadas de las secciones.

    Returns:
    tuple: Coordenadas de la sección que contiene la columna, o None si no se encuentra.
    """
    x1, y1, x2, y2 = column_box
    x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
    
    for sx1, sy1, sx2, sy2 in section_boxes:
        if (sx1 <= x_center <= sx2) and (sy1 <= y_center <= sy2):
            return (sx1, sy1, sx2, sy2)
    return None

def get_horizontal_lines(boxes, classes, names):
    """
    Extrae las coordenadas Y de las líneas horizontales (bordes de página, sección y encabezado).

    Args:
    boxes (np.array): Array de cajas delimitadoras.
    classes (np.array): Array de clases correspondientes a cada caja.
    names (dict): Diccionario que mapea índices de clase a nombres de clase.

    Returns:
    list: Lista ordenada de coordenadas Y únicas de las líneas horizontales.
    """
    horizontal_lines = []
    for box, cls in zip(boxes, classes):
        class_name = names[int(cls)]
        if class_name in ['pagina', 'seccion', 'encabezado']:
            x1, y1, x2, y2 = map(int, box.tolist())
            horizontal_lines.extend([y1, y2])
    return sorted(set(horizontal_lines))

def is_overlapping_header(column_box, header_boxes):
    """
    Verifica si una columna se superpone con algún encabezado.

    Args:
    column_box (list): Coordenadas [x1, y1, x2, y2] de la columna.
    header_boxes (list): Lista de coordenadas de los encabezados.

    Returns:
    tuple: (bool, int) Indica si hay superposición y la coordenada Y inferior del encabezado.
    """
    x1, y1, x2, y2 = column_box
    for hx1, hy1, hx2, hy2 in header_boxes:
        if not (x2 < hx1 or x1 > hx2):
            return True, hy2
    return False, None

def adjust_column_boundaries(column_box, horizontal_lines, header_boxes, section_boxes):
    """
    Ajusta los límites de una columna basándose en las líneas horizontales, encabezados y secciones.

    Args:
    column_box (np.array): Coordenadas [x1, y1, x2, y2] de la columna.
    horizontal_lines (list): Lista de coordenadas Y de líneas horizontales.
    header_boxes (list): Lista de coordenadas de los encabezados.
    section_boxes (list): Lista de coordenadas de las secciones.

    Returns:
    list: Coordenadas ajustadas [x1, y1, x2, y2] de la columna.
    """
    x1, y1, x2, y2 = map(int, column_box.tolist())
    
    containing_section = get_containing_section([x1, y1, x2, y2], section_boxes)
    
    if containing_section:
        _, section_top, _, section_bottom = containing_section
        horizontal_lines = [y for y in horizontal_lines if section_top <= y <= section_bottom]
    
    overlaps_header, header_bottom = is_overlapping_header([x1, y1, x2, y2], header_boxes)
    
    if overlaps_header:
        new_y1 = header_bottom
    else:
        upper_lines = [y for y in horizontal_lines if y < y1]
        new_y1 = max(upper_lines) if upper_lines else y1
    
    lower_lines = [y for y in horizontal_lines if y > y2]
    new_y2 = min(lower_lines) if lower_lines else y2
    
    if containing_section:
        _, section_top, _, section_bottom = containing_section
        new_y1 = max(new_y1, section_top)
        new_y2 = min(new_y2, section_bottom)
    
    return [x1, new_y1, x2, new_y2]

def calculate_overlap_vectorized(box, boxes):
    """
    Calcula el porcentaje de superposición entre una caja y un conjunto de cajas de manera vectorizada.

    Args:
    box (np.array): Coordenadas [x1, y1, x2, y2] de la caja de referencia.
    boxes (np.array): Array de coordenadas de las cajas a comparar.

    Returns:
    np.array: Array de porcentajes de superposición.
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    overlap_percentage = intersection_area / np.minimum(box_area, boxes_area)
    return overlap_percentage

def remove_overlapping_columns(adjusted_boxes, original_boxes):
    """
    Elimina columnas que se superponen significativamente.

    Args:
    adjusted_boxes (list): Lista de coordenadas de columnas ajustadas.
    original_boxes (list): Lista de coordenadas de columnas originales.

    Returns:
    list: Lista de tuplas (caja_original, caja_ajustada) de las columnas no superpuestas.
    """
    adjusted_boxes = np.array(adjusted_boxes)
    original_boxes = np.array(original_boxes)
    
    areas = (adjusted_boxes[:, 2] - adjusted_boxes[:, 0]) * (adjusted_boxes[:, 3] - adjusted_boxes[:, 1])
    
    overlaps = np.zeros((len(adjusted_boxes), len(adjusted_boxes)))
    for i in range(len(adjusted_boxes)):
        overlaps[i] = calculate_overlap_vectorized(adjusted_boxes[i], adjusted_boxes)
    
    boxes_to_remove = set()
    for i in range(len(adjusted_boxes)):
        if i in boxes_to_remove:
            continue
        
        overlap_indices = np.where(overlaps[i] >= 0.9)[0]
        overlap_indices = overlap_indices[overlap_indices > i]
        
        if len(overlap_indices) > 0:
            heights = adjusted_boxes[overlap_indices, 3] - adjusted_boxes[overlap_indices, 1]
            if heights.max() > (adjusted_boxes[i, 3] - adjusted_boxes[i, 1]):
                boxes_to_remove.add(i)
            else:
                boxes_to_remove.update(overlap_indices)
    
    keep_indices = list(set(range(len(adjusted_boxes))) - boxes_to_remove)
    return list(zip(original_boxes[keep_indices], adjusted_boxes[keep_indices]))

def sort_columns_by_section(columns, sections):
    """
    Ordena las columnas por sección y de izquierda a derecha dentro de cada sección.

    Args:
    columns (list): Lista de coordenadas de columnas.
    sections (list): Lista de coordenadas de secciones.

    Returns:
    list: Lista de diccionarios, cada uno con una sección y sus columnas ordenadas.
    """
    if not sections:
        return [{'box': [0, 0, float('inf'), float('inf')], 'columns': sorted(columns, key=lambda c: c[0])}]

    sorted_sections = sorted(sections, key=lambda s: s[1])
    result = []
    for section in sorted_sections:
        section_columns = [col for col in columns if col[1] >= section[1] and col[3] <= section[3]]
        result.append({
            'box': section,
            'columns': sorted(section_columns, key=lambda c: c[0])
        })
    return result

def cut_and_save_columns(image, sorted_sections, input_file, output_dir):
    """
    Recorta y guarda las columnas como imágenes separadas.

    Args:
    image (np.array): Imagen original.
    sorted_sections (list): Lista de secciones con sus columnas ordenadas.
    input_file (str): Ruta del archivo de imagen de entrada.
    output_dir (str): Directorio donde se guardarán las imágenes recortadas.
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    margin = 5  # Margen de 5 píxeles en todos los lados

    for s_idx, section in enumerate(sorted_sections):
        for c_idx, column in enumerate(section['columns']):
            x1, y1, x2, y2 = map(int, column)
            
            # Añadir margen y asegurarse de que no exceda los límites de la imagen
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.shape[1], x2 + margin)
            y2 = min(image.shape[0], y2 + margin)
            
            cropped_image = image[y1:y2, x1:x2]
            file_name = f"{base_name}_secc_{s_idx:02d}_col_{c_idx+1:02d}.jpg"
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, cropped_image)

def process_image(image_path, model):
    """
    Procesa una imagen individual, detectando y ajustando las columnas.

    Args:
    image_path (str): Ruta de la imagen a procesar.
    model (YOLO): Modelo YOLO cargado para la detección.

    Returns:
    tuple: (imagen_procesada, secciones_ordenadas)
    """
    image = cv2.imread(image_path)
    results = model.predict(image_path)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names

    horizontal_lines = get_horizontal_lines(boxes, classes, names)
    header_boxes = get_header_boundaries(boxes, classes, names)
    section_boxes = get_section_boundaries(boxes, classes, names)

    original_column_boxes = []
    adjusted_column_boxes = []

    for box, cls in zip(boxes, classes):
        class_name = names[int(cls)]
        if class_name == 'columna':
            original_column_boxes.append(box)
            adjusted_box = adjust_column_boundaries(box, horizontal_lines, header_boxes, section_boxes)
            adjusted_column_boxes.append(adjusted_box)

    kept_boxes = remove_overlapping_columns(adjusted_column_boxes, original_column_boxes)
    sorted_sections = sort_columns_by_section([box for _, box in kept_boxes], section_boxes)
    
    return image, sorted_sections

def process_directory(input_dir, output_dir, model):
    """
    Procesa todas las imágenes en un directorio dado.

    Args:
    input_dir (str): Directorio que contiene las imágenes a procesar.
    output_dir (str): Directorio donde se guardarán las columnas recortadas.
    model (YOLO): Modelo YOLO cargado para la detección.
    """
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            
            try:
                processed_image, sorted_sections = process_image(input_path, model)
                cut_and_save_columns(processed_image, sorted_sections, input_path, output_dir)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Cargar el modelo
model = YOLO("./modelo/yolo11x-layout.pt")

# Directorios de entrada y salida
input_directory = "input"
output_directory = "output"

# Procesar todas las imágenes en el directorio de entrada
process_directory(input_directory, output_directory, model)
