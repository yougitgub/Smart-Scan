if __name__ == "__main__":
    from gui import *
    from generate_face_embeddings import *
    from face_recognition import *
    if not os.path.exists(KNOWN_EMBEDDINGS_FILE):
        print("No known embeddings found. Generating from enrollment images...")
        generate_known_embeddings(ENROLLMENT_DIRECTORY)
    root.mainloop()
    real_time_face_recognition()  # Start the main face recognition function
    check_absence()  # Check and mark absences in the Excel sheet