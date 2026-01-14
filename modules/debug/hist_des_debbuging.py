import cv2
import numpy as np

def draw_hist_graph(hist, color=(0, 255, 0), size=(200, 100), bg_color=(0, 0, 0)):
    canvas = np.full((size[1], size[0], 3), bg_color, dtype="uint8")
    if hist is None: return canvas

    disp_hist = hist.copy()
    cv2.normalize(disp_hist, disp_hist, alpha=0, beta=size[1], norm_type=cv2.NORM_MINMAX)

    bin_w = max(1, int(size[0] / len(hist)))
    
    for i in range(len(hist)):
        val = int(disp_hist[i])
        cv2.rectangle(canvas, (i * bin_w, size[1] - val), 
                      ((i + 1) * bin_w, size[1]), color, -1)
    return canvas

def visualize_prediction_details(classifier, roi, mask, predicted_name, wait_time=0):
    if roi is None or classifier is None: return

    h, w = roi.shape[:2]
    
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kp = classifier.orb.detect(gray, mask=mask)
    roi_with_kp = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    curr_hist = classifier._calc_histogram(roi, mask)
    curr_avg_color = classifier._calc_average_color(roi, mask)
    
    graph_curr = draw_hist_graph(curr_hist, color=(0, 255, 0))
    cv2.putText(graph_curr, "Current Hist", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    graph_ref = np.zeros_like(graph_curr)
    ref_info_text = "Ref: None"
    
    score_lab = 0.0
    score_hist = 1.0
    
    if predicted_name != "Unknown" and predicted_name in classifier.references:
        ref_data = classifier.references[predicted_name]
        
        graph_ref = draw_hist_graph(ref_data['hist'], color=(0, 255, 255))
        dist_lab = np.linalg.norm(curr_avg_color - ref_data['color'])
        dist_hist = cv2.compareHist(curr_hist, ref_data['hist'], cv2.HISTCMP_BHATTACHARYYA)
        
        ref_info_text = f"Ref: {predicted_name}"
        score_text = f"LabDist: {dist_lab:.1f} | HistDiff: {dist_hist:.2f}"
    else:
        score_text = "Unknown / No Match"

    cv2.putText(graph_ref, ref_info_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    img_h = 120
    img_w = 120
    img1 = cv2.resize(masked_roi, (img_w, img_h))
    img2 = cv2.resize(roi_with_kp, (img_w, img_h))
    col_left = np.vstack([img1, img2])

    g_h, g_w = 120, 200
    graph_curr = cv2.resize(graph_curr, (g_w, g_h))
    graph_ref = cv2.resize(graph_ref, (g_w, g_h))
    col_right = np.vstack([graph_curr, graph_ref])

    debug_panel = np.hstack([col_left, col_right])
    info_bar = np.zeros((40, debug_panel.shape[1], 3), dtype="uint8")
    cv2.putText(info_bar, f"Pred: {predicted_name}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(info_bar, score_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    final_img = np.vstack([debug_panel, info_bar])

    cv2.imshow("Debug: Token Analysis", final_img)
    if wait_time > 0:
        cv2.waitKey(wait_time)


def generate_debug_image(classifier, roi, mask, predicted_name, circle_id):
    """
    ROI, Keypoints, Histograms for debugging
    Now histograms don't take part in class recognition
    """
    if roi is None or classifier is None: return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kp = classifier.orb.detect(gray, mask=mask)
    roi_with_kp = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)

    curr_hist = classifier._calc_histogram(roi, mask)
    curr_avg_color = classifier._calc_average_color(roi, mask)
    graph_curr = draw_hist_graph(curr_hist, color=(0, 255, 0), size=(150, 80))
    graph_ref = np.zeros_like(graph_curr)
    info_text = "?"
    
    if predicted_name != "Unknown" and predicted_name in classifier.references:
        ref_data = classifier.references[predicted_name]
        graph_ref = draw_hist_graph(ref_data['hist'], color=(0, 255, 255), size=(150, 80))
        
        dist_lab = np.linalg.norm(curr_avg_color - ref_data['color'])
        dist_hist = cv2.compareHist(curr_hist, ref_data['hist'], cv2.HISTCMP_BHATTACHARYYA)
        info_text = f"L:{dist_lab:.0f} H:{dist_hist:.2f}"
    
    H, W = 80, 80 
    img_tl = cv2.resize(masked_roi, (W, H))
    img_tr = cv2.resize(roi_with_kp, (W, H))
    graph_curr = cv2.resize(graph_curr, (W*2, H))
    graph_ref = cv2.resize(graph_ref, (W*2, H))
    
    row_top = np.hstack([img_tl, img_tr])
    panel = np.vstack([row_top, graph_curr, graph_ref])
    header = np.zeros((30, panel.shape[1], 3), dtype="uint8")
    label = f"#{circle_id} {predicted_name}"
    color = (0, 255, 0) if predicted_name != "Unknown" else (0, 0, 255)
    cv2.putText(header, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(header, info_text, (panel.shape[1]-80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return np.vstack([header, panel])    
        