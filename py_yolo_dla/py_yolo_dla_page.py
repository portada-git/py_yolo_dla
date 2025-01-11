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

import os  # Para operaciones de sistema de archivos

import cv2  # Para procesamiento de imágenes
import numpy as np  # Para operaciones numéricas eficientes
# Importaciones necesarias
from ultralytics import YOLO  # Para cargar y usar el modelo YOLO


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
    # max_page_area = 0
    # max_page = None
    for box, cls in zip(boxes, classes):
        class_name = names[int(cls)]
        if class_name == 'seccion':
            x1, y1, x2, y2 = map(int, box.tolist())
            section_boxes.append([x1, y1, x2, y2])

    return section_boxes


def get_other_boundaries(boxes, classes, names, guess_page):
    other_boxes = []
    max_page_area = 0
    max_page = None
    for box, cls in zip(boxes, classes):
        class_name = names[int(cls)]
        if class_name == 'bloque':
            x1, y1, x2, y2 = map(int, box.tolist())
            other_boxes.append([x1, y1, x2, y2])
        elif class_name == 'objeto':
            x1, y1, x2, y2 = map(int, box.tolist())
            other_boxes.append([x1, y1, x2, y2])
        # elif class_name == 'pagina':
        #     x1, y1, x2, y2 = map(int, box.tolist())
        #     parea = (x2 - x1) * (y2 - y1)
        #     if parea > max_page_area:
        #         if max_page is not None:
        #             other_boxes.append(max_page)
        #         max_page_area = parea
        #         max_page = [x1, y1, x2, y2]
        #     else:
        #         other_boxes.append([x1, y1, x2, y2])
    # if max_page is not None:
    #     intersection = max(0, min(max_page[2], guess_page[2]) - max(max_page[0], guess_page[0])) * max(0, min(max_page[3],guess_page[3]) - max(max_page[1],guess_page[1]))
    #     guess_area = (guess_page[2] - guess_page[0]) * (guess_page[3] - guess_page[1])
    #     union = max_page_area + guess_area - intersection
    #     iou = intersection / union if union > 0 else 0
    #     if iou < 0.80:
    #         other_boxes.append(max_page)

    return other_boxes


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
    original_boxes = np.array(original_boxes)
    adjusted_boxes = np.array(adjusted_boxes)
    keep_indices = _get_non_overlapping_indexes(adjusted_boxes)
    return list(zip(original_boxes[keep_indices], adjusted_boxes[keep_indices]))


def remove_overlapping_boxes(boxes):
    boxes = np.array(boxes)
    keep_indices = _get_non_overlapping_indexes(boxes)
    return list(boxes[keep_indices])


def _get_non_overlapping_indexes(boxes):
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    overlaps = np.zeros((len(boxes), len(boxes)))
    for i in range(len(boxes)):
        overlaps[i] = calculate_overlap_vectorized(boxes[i], boxes)

    boxes_to_remove = set()
    for i in range(len(boxes)):
        if i in boxes_to_remove:
            continue

        overlap_indices = np.where(overlaps[i] >= 0.85)[0]
        overlap_indices = overlap_indices[overlap_indices > i]

        if len(overlap_indices) > 0:
            heights = boxes[overlap_indices, 3] - boxes[overlap_indices, 1]
            if heights.max() > (boxes[i, 3] - boxes[i, 1]):
                boxes_to_remove.add(i)
            else:
                boxes_to_remove.update(overlap_indices)

    keep_indices = list(set(range(len(boxes))) - boxes_to_remove)
    return keep_indices


def sort_columns_by_section(columns, sections, guess_page=None):
    """
    Ordena las columnas por sección y de izquierda a derecha dentro de cada sección.

    Args:
    columns (list): Lista de coordenadas de columnas.
    sections (list): Lista de coordenadas de secciones.

    Returns:
    list: Lista de diccionarios, cada uno con una sección y sus columnas ordenadas.
    """
    if guess_page is None:
        guess_page = [0, 0, float('inf'), float('inf')]
    if not sections:
        return [{'box': [guess_page[0], guess_page[1], guess_page[2], guess_page[3]],
                 'columns': sorted(columns, key=lambda c: c[0])}]

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
    os.makedirs(output_dir, exist_ok=True)

    ret = cut_columns_as_json(image, sorted_sections, input_file)
    for item in ret:
        output_path = os.path.join(output_dir, item["file_name"])
        cv2.imwrite(output_path, item["image"])


def redefine_sections(sections):
    to_remove = set()
    result = []
    for i, section in enumerate(sections):
        if i in to_remove:
            continue

        s_center = (section["box"][0] + section["box"][2]) / 2
        for j in range(i + 1, len(sections)):
            sj_center = (sections[j]["box"][0] + sections[j]["box"][2]) / 2
            if abs(s_center - sj_center) < 60 and len(section["columns"]) == len(sections[j]["columns"]) and \
                    sections[j]['box'][1] - section['box'][3] < 60:  # threshold == 60
                to_remove.add(j)
                section['box'][3] = max(section['box'][3], sections[j]['box'][3])
                for c in range(len(section['columns'])):
                    section['columns'][c][3] = max(section['columns'][c][3], sections[j]['columns'][c][3])
        result.append(section)

    return result


def get_model(fpath=None):
    if fpath is None:
        p = os.path.abspath(os.path.dirname(__file__))
        fpath = f"{p}/modelo/yolo11x-layout.pt"
    return YOLO(fpath)


def cut_columns_as_json(image, sorted_sections, input_file):
    """
    Recorta y guarda las columnas como imágenes separadas.

    Args:
    image (np.array): Imagen original.
    sorted_sections (list): Lista de secciones con sus columnas ordenadas.
    input_file (str): Ruta del archivo de imagen de entrada.
    output_dir (str): Directorio donde se guardarán las imágenes recortadas.
    """

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    margin = 5  # Margen de 5 píxeles en todos los lados

    ret = []
    for s_idx, section in enumerate(sorted_sections):
        for c_idx, column in enumerate(section['columns']):
            x1, y1, x2, y2 = map(int, column)

            # Añadir margen y asegurarse de que no exceda los límites de la imagen
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.shape[1], x2 + margin)
            y2 = min(image.shape[0], y2 + margin)

            cropped_image = image[y1:y2, x1:x2]
            id = f"{s_idx:02d}_{c_idx + 1:02d}"
            file_name = f"{base_name}_secc_{s_idx:02d}_col_{c_idx + 1:02d}.jpg"
            ret.append({"id": id, "file_name": file_name, "image": cropped_image})
    return ret


def process_image_from_path(image_path, model=None):
    if model is None:
        model = get_model()
    image = cv2.imread(image_path)
    sorted_sections = process_image(image, model)
    return image, sorted_sections


def process_image(image: np.array, model=None):
    """
    Procesa una imagen individual, detectando y ajustando las columnas.

    Args:
    image (np.array): Ruta de la imagen a procesar.
    model (YOLO): Modelo YOLO cargado para la detección.

    Returns:
    tuple: (imagen_procesada, secciones_ordenadas)
    """
    if model is None:
        model = get_model()
    results = model.predict(image)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names

    guess_page = [np.min(boxes[:, 0]), np.min(boxes[:, 1]), np.max(boxes[:, 2]), np.max(boxes[:, 3])]

    horizontal_lines = get_horizontal_lines(boxes, classes, names)
    header_boxes = get_header_boundaries(boxes, classes, names)
    section_boxes = get_section_boundaries(boxes, classes, names)
    other_boxes = get_other_boundaries(boxes, classes, names, guess_page)

    # if len(section_boxes) > 0:
    #     section_boxes = remove_overlapping_boxes(section_boxes)

    original_column_boxes = []
    adjusted_column_boxes = []

    for box, cls in zip(boxes, classes):
        class_name = names[int(cls)]
        if class_name == 'columna':
            original_column_boxes.append(box)
            adjusted_box = adjust_column_boundaries(box, horizontal_lines, header_boxes, section_boxes)
            adjusted_column_boxes.append(adjusted_box)

    if len(original_column_boxes) > 1:
        kept_boxes = remove_overlapping_columns(adjusted_column_boxes, original_column_boxes)
    else:
        kept_boxes = []
    sorted_sections = sort_columns_by_section([box for _, box in kept_boxes], section_boxes, guess_page)

    kept_others = []
    to_delete = []
    for other in other_boxes:
        keep = True
        reduce_columns = []
        area_other = (other[2] - other[0]) * (other[3] - other[1])
        for s, section in enumerate(sorted_sections):
            for c, col in enumerate(section["columns"]):
                p_intersection = max(0, min(other[2], col[2]) - max(other[0], col[0])) * max(0, min(other[3],
                                                                                                    col[3]) - max(
                    other[1], col[1])) / area_other
                if p_intersection > 0.65:
                    keep = False
                    break
                p_lx = (min(col[2], other[2]) - max(col[0], other[0])) / (col[2] - col[0])
                m_y1 = max(col[1], other[1])
                m_y2 = min(col[3], other[3])
                if (m_y2 - m_y1 > 0) and (p_lx > 0.75 or len(reduce_columns) > 1 and p_lx > 0.25):
                    reduce_columns.append([s, c])
            if not keep:
                break
        if keep:
            # kept_others.append({'box': other, 'columns': [other]})
            kept_others.append({'box': other, 'columns': []})
            for index in reduce_columns:
                cy1 = sorted_sections[index[0]]["columns"][index[1]][1]
                cy2 = sorted_sections[index[0]]["columns"][index[1]][3]
                oy1 = other[1]
                oy2 = other[3]
                o_d = oy2 - oy1
                if oy2 - cy1 - o_d > 25 >= cy2 - oy1 - o_d:
                    sorted_sections[index[0]]["columns"][index[1]][3] = oy1
                    if sorted_sections[index[0]]["columns"][index[1]][3] - \
                            sorted_sections[index[0]]["columns"][index[1]][1] < 25:
                        to_delete.append(index)
                elif cy2 - oy1 - o_d > 25 >= oy2 - cy1 - o_d:
                    sorted_sections[index[0]]["columns"][index[1]][1] = oy2
                    if sorted_sections[index[0]]["columns"][index[1]][3] - \
                            sorted_sections[index[0]]["columns"][index[1]][1] < 25:
                        to_delete.append(index)
                else:
                    x1 = sorted_sections[index[0]]["columns"][index[1]][0]
                    y1 = sorted_sections[index[0]]["columns"][index[1]][1]
                    x2 = sorted_sections[index[0]]["columns"][index[1]][2]
                    y2 = sorted_sections[index[0]]["columns"][index[1]][3]
                    sorted_sections[index[0]]["columns"][index[1]][3] = oy1
                    if sorted_sections[index[0]]["columns"][index[1]][3] - \
                            sorted_sections[index[0]]["columns"][index[1]][1] < 25:
                        to_delete.append(index)
                    if y2 - oy2 >= 25:
                        sorted_sections[index[0]]["columns"].append([x1, oy2, x2, y2])
    for index in to_delete:
        sorted_sections[index[0]]["columns"].pop(index[1])
    for section in sorted_sections:
        section["columns"].sort(key=lambda X: X[0] * 10000 + X[1])

    # kept_others.sort(key=lambda X:X["box"][1] * 10000 + X["box"][0])
    sorted_sections.extend(kept_others)
    sorted_sections.sort(key=lambda X: X["box"][1] * 10000 + X["box"][0])
    sorted_sections = redefine_sections(sorted_sections)

    return sorted_sections


def process_directory(input_dir, output_dir, model=None):
    """
    Procesa todas las imágenes en un directorio dado.

    Args:
    input_dir (str): Directorio que contiene las imágenes a procesar.
    output_dir (str): Directorio donde se guardarán las columnas recortadas.
    model (YOLO): Modelo YOLO cargado para la detección.
    """
    if model is None:
        model = get_model()

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)

            try:
                processed_image, sorted_sections = process_image_from_path(input_path, model)
                cut_and_save_columns(processed_image, sorted_sections, input_path, output_dir)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


def main():
    # Directorios de entrada y salida
    input_directory = "input"
    output_directory = "output"

    # Procesar todas las imágenes en el directorio de entrada
    process_directory(input_directory, output_directory)


if __name__ == "__main__":
    main()
