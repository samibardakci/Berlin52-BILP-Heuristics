import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

def load_distance_matrix(file_path: str) -> np.ndarray:
    """
    Loads a TSP distance matrix from a file.
    Supports:
      - .xml: TSPLIB XML format (vertex/edge tags)
      - .csv: Matrix format or Edge list format (columns: source, target, cost)
      - .xlsx: Matrix format or Edge list format
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.xml':
        return _parse_xml(file_path)
    elif ext == '.csv':
        return _parse_csv(file_path)
    elif ext in ['.xlsx', '.xls']:
        return _parse_excel(file_path)
    elif ext in ['.txt', '.tsp']:
        return _parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def _parse_txt(file_path: str) -> np.ndarray:
    coords = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    in_coord_section = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("NODE_COORD_SECTION"):
            in_coord_section = True
            continue
        elif line.startswith("EOF") or line == "EOF":
            break
            
        if in_coord_section:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    # Format: NodeID X Y
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append((x, y))
                except ValueError:
                    pass
                    
    if not coords:
        raise ValueError("No coordinates found in TSP TXT file.")
        
    coords = np.array(coords)
    n = len(coords)
    matrix = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0
            else:
                xd = coords[i, 0] - coords[j, 0]
                yd = coords[i, 1] - coords[j, 1]
                # TSPLIB EUC_2D distance calculation (rounded to nearest integer)
                dist = np.sqrt(xd**2 + yd**2)
                matrix[i, j] = float(round(dist))
                
    return matrix


def _parse_xml(file_path: str) -> np.ndarray:
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    graph = root.find('graph')
    if graph is None:
        raise ValueError("Invalid TSP XML format: '<graph>' tag not found.")
        
    vertices = graph.findall('vertex')
    num_vertices = len(vertices)
    if num_vertices == 0:
        raise ValueError("No vertices found in XML.")
        
    matrix = np.zeros((num_vertices, num_vertices), dtype=float)
    
    for source_idx, vertex in enumerate(vertices):
        for edge in vertex.findall('edge'):
            cost = float(edge.attrib['cost'])
            target_idx = int(edge.text)
            matrix[source_idx, target_idx] = cost
            
    return matrix

def _parse_csv(file_path: str) -> np.ndarray:
    # Read first line to detect format
    with open(file_path, 'r') as f:
        first_line = f.readline().strip().lower()
        
    # Check if it has edge list headers
    if 'source' in first_line and 'target' in first_line and 'cost' in first_line:
        df = pd.read_csv(file_path)
        return _build_matrix_from_edge_list(df)
        
    # Otherwise read as matrix
    df = pd.read_csv(file_path, header=None)
    # Check if first row is a header like "0,1,2,...,N-1"
    first_row = df.iloc[0].values
    try:
        # If all elements in first row can be converted to int and match 0..N-1, it is a header
        if len(first_row) > 1 and all(float(val).is_integer() and int(float(val)) == idx for idx, val in enumerate(first_row)):
            df = pd.read_csv(file_path)
    except (ValueError, TypeError):
        pass
        
    return df.to_numpy(dtype=float)

def _parse_excel(file_path: str) -> np.ndarray:
    # Read the first sheet
    df = pd.read_excel(file_path)
    
    # Check if it is an edge list
    cols = [str(c).lower().strip() for c in df.columns]
    if 'source' in cols and 'target' in cols and 'cost' in cols:
        # Standardize column names
        df.columns = cols
        return _build_matrix_from_edge_list(df)
        
    # Otherwise it is a matrix
    # Excel reader parses headers automatically. If the header is numeric, it is a matrix column label.
    # We convert the dataframe directly to numpy.
    return df.to_numpy(dtype=float)

def _build_matrix_from_edge_list(df: pd.DataFrame) -> np.ndarray:
    # Ensure columns exist
    required = {'source', 'target', 'cost'}
    cols = {str(c).lower().strip() for c in df.columns}
    if not required.issubset(cols):
        raise ValueError(f"Edge list format requires columns: source, target, cost. Found: {df.columns.tolist()}")
        
    # Standardize column names
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    num_vertices = int(max(df['source'].max(), df['target'].max())) + 1
    matrix = np.zeros((num_vertices, num_vertices), dtype=float)
    
    for _, row in df.iterrows():
        src = int(row['source'])
        tgt = int(row['target'])
        cost = float(row['cost'])
        matrix[src, tgt] = cost
        
    return matrix

if __name__ == '__main__':
    # Test script locally
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        m_xml = load_distance_matrix(os.path.join(base_dir, 'berlin52.xml'))
        m_csv_edges = load_distance_matrix(os.path.join(base_dir, 'berlin52_edges.csv'))
        m_csv_matrix = load_distance_matrix(os.path.join(base_dir, 'berlin52_matrix.csv'))
        m_txt = load_distance_matrix(os.path.join(base_dir, 'berlin52-tsp.txt'))
        
        print("XML Matrix Shape:", m_xml.shape)
        print("CSV Edges Matrix Shape:", m_csv_edges.shape)
        print("CSV Matrix Shape:", m_csv_matrix.shape)
        print("TXT Matrix Shape:", m_txt.shape)
        
        # Verify XML vs CSVs
        diff1 = np.abs(m_xml - m_csv_edges).max()
        diff2 = np.abs(m_xml - m_csv_matrix).max()
        print(f"Difference XML vs Edges: {diff1}")
        print(f"Difference XML vs Matrix: {diff2}")
        
        # Print differences between XML (original) and TXT (coordinate based)
        diff_txt = np.abs(m_xml - m_txt)
        print(f"Max Difference XML vs TXT: {diff_txt.max()}")
        print(f"Number of differing entries: {np.sum(diff_txt > 0.0)}")
        
    except Exception as e:
        print("Error during test:", e)

