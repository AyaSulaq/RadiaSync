import React, { useEffect } from 'react';

const Modal = ({ open, onClose, images }) => {
  // Close the modal when clicking outside of it
  useEffect(() => {
    const handleOutsideClick = (event) => {
      if (event.target.classList.contains('modal-overlay')) {
        onClose();
      }
    };

    if (open) {
      document.addEventListener('click', handleOutsideClick);
    }

    return () => {
      document.removeEventListener('click', handleOutsideClick);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="modal-overlay fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="relative w-full max-w-3xl mx-auto my-6 bg-white p-4 rounded shadow-lg">
        <div className="flex items-start justify-between border-b border-gray-300 pb-4 mb-4">
          <h3 className="text-2xl font-semibold">Image Preview</h3>
          <button
            className="text-black text-3xl font-semibold"
            onClick={onClose}
          >
            &times;
          </button>
        </div>
        <div className="flex justify-between gap-4">
          {images.map((imageUrl, index) => (
            <div key={index} className="w-1/2 text-center">
              <div className="mb-2 font-semibold">
                {index === 0 ? 'Original' : 'Output'}
              </div>
              <img src={imageUrl} alt={`Image ${index}`} className="max-w-full max-h-full" />
            </div>
          ))}
        </div>
        <div className="flex justify-end mt-4">
          <button
            className="text-blue-600 hover:text-blue-800 font-semibold"
            onClick={onClose}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default Modal;
