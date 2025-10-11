if __name__ == "__main__":
    from gui import *
    from generate_face_embeddings import *
    from face_recognition import *
    if not os.path.exists(KNOWN_EMBEDDINGS_FILE):
        print("No known embeddings found. Generating from enrollment images...")
        generate_known_embeddings(ENROLLMENT_DIRECTORY)

    root.mainloop()
    try:
        import openpyxl
        workbook = openpyxl.load_workbook('DashBoard.xlsx')
        sheet = workbook.active
        print("Excel file loaded successfully.")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        exit()
    real_time_face_recognition()  # Start the main face recognition function
