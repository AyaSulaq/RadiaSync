import React, { useState, useEffect } from 'react';
import moment from 'moment';
import Cookies from 'js-cookie';

// Modal component
const Modal = ({ open, onClose, images }) => {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-x-hidden overflow-y-auto outline-none focus:outline-none">
      <div className="relative w-auto max-w-3xl mx-auto my-6">
        <div className="relative flex flex-col w-full bg-white border-0 rounded-lg shadow-lg outline-none focus:outline-none">
          <div className="flex items-start justify-between p-5 border-b border-solid rounded-t border-gray-300">
            <h3 className="text-3xl font-semibold">Image Preview</h3>
            <button
              className="p-1 ml-auto bg-transparent border-0 text-black opacity-5 float-right text-3xl leading-none font-semibold outline-none focus:outline-none"
              onClick={onClose} // Call onClose when close button is clicked
            >
              <span className="bg-transparent text-black opacity-5 h-6 w-6 text-2xl block outline-none focus:outline-none">
                ×
              </span>
            </button>
          </div>
          <div className="p-6 flex-auto overflow-y-auto">
            {images.map((imageUrl, index) => (
              <div key={index} className="mb-4">
                <img src={imageUrl} alt={`Image ${index}`} className="max-w-full max-h-full" />
              </div>
            ))}
          </div>
          <div className="flex items-center justify-end p-6 border-t border-solid rounded-b border-gray-300">
            <button
              className="text-blue-600 hover:text-blue-800 font-semibold mr-4"
              onClick={onClose} // Call onClose when close button is clicked
            >
              Close
            </button>
          </div>
        </div>
      </div>
      <div className="fixed inset-0 z-40 bg-black opacity-25"></div>
    </div>
  );
};

export default function UserDashboard() {
  const [workContents, setWorkContents] = useState([]);
  const [modalImages, setModalImages] = useState([]);
  const [openModal, setOpenModal] = useState(false);

  useEffect(() => {
    // Fetch work contents data from the API
    fetchWorkContents();
  }, []);

  const userID = Cookies.get('userId');
  const fetchWorkContents = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/WorkContents/user/'+userID, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });
      const data = await response.json();
      if (data.success) {
        setWorkContents(data.data);
      } else {
        console.error('Failed to fetch work contents:', data.message);
      }
    } catch (error) {
      console.error('Error fetching work contents:', error);
    }
  };

  const handleViewResult = async (id) => {
    try {
      const response = await fetch(`http://localhost:5000/api/WorkContents/${id}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });

      const data = await response.json();
      if (data.success) {
        // Extract image URLs from the response data
        const imageUrls = data.data.map(imageData => imageData);
        // Pass image URLs to the Modal component
        setModalImages(imageUrls);
        setOpenModal(true);
      } else {
        console.error('Failed to fetch images:', data.message);
      }
    } catch (error) {
      console.error('Error fetching images:', error);
    }
  };

  // Function to handle viewing patient profile
  const handleViewPatientProfile = (patientId) => {
    // Implement functionality to view patient profile based on patientId
    console.log(`Viewing patient profile for patientId: ${patientId}`);
  };

  // Function to close modal
  const handleCloseModal = () => {
    setOpenModal(false);
  };

  return (
    <div className="container mx-auto mt-8">
      <h1 className="text-3xl font-bold mb-4">Dashboard</h1>
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white border border-gray-200">
          <thead>
            <tr className="bg-gray-100 border-b border-gray-200">
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">File Name</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Creation Date</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Finished Date</th>
              <th className="px-6 py-3"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {workContents.map(workContent => (
              <tr key={workContent.id}>
                <td className="px-6 py-4 whitespace-nowrap">{workContent.fileName}</td>
                <td className="px-6 py-4 whitespace-nowrap">{moment(workContent.createdAt).format('YYYY-MM-DD HH:mm:ss')}</td>
                <td className="px-6 py-4 whitespace-nowrap">{workContent.finishedAt ? moment(workContent.finishedAt).format('YYYY-MM-DD HH:mm:ss') : 'N/A'}</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <button className="text-blue-600 hover:text-blue-800 font-semibold mr-4" onClick={() => handleViewResult(workContent.id)}>View Result</button>
                  <button className="text-blue-600 hover:text-blue-800 font-semibold" onClick={() => handleViewPatientProfile(workContent.patientId)}>View Patient Profile</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* Modal component */}
      <Modal open={openModal} onClose={handleCloseModal} images={modalImages} />
    </div>
  );
}
