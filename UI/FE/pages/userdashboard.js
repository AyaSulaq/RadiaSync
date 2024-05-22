import React, { useState, useEffect } from 'react';
import moment from 'moment';
import Cookies from 'js-cookie';
import Modal from './Modal';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

export default function UserDashboard() {
  const [workContents, setWorkContents] = useState([]);
  const [modalImages, setModalImages] = useState([]);
  const [openModal, setOpenModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [cardStatus, setCardStatus] = useState({});

  const fetchWorkContents = async (userId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/WorkContents/byPatient/${userId}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });
      const data = await response.json();
      if (data.success) {
        setWorkContents(data.data);
        setCardStatus((prevStatus) => ({ ...prevStatus, [userId]: 'success' }));
      } else {
        setCardStatus((prevStatus) => ({ ...prevStatus, [userId]: 'error' }));
      }
    } catch (error) {
      setCardStatus((prevStatus) => ({ ...prevStatus, [userId]: 'error' }));
      console.error('Error fetching work contents:', error);
    }
  };

  const handleSearch = async (event) => {
    if (event.key === 'Enter') {
      setLoading(true);
      // Clear previous search results and card statuses
      setSearchResults([]);
      setCardStatus({});
      try {
        const response = await fetch(`http://localhost:5000/api/Users/search?query=${searchQuery}`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json'
          }
        });
        const data = await response.json();
        if (data.success) {
          toast.success(data.message);
          setSearchResults(data.data);
        } else {
          toast.error(data.message);
          console.error('Failed to fetch users:', data.message);
        }
      } catch (error) {
        console.error('Error fetching users:', error);
      } finally {
        setLoading(false);
      }
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
        const imageUrls = data.data;
        setModalImages(imageUrls);
        setOpenModal(true);
      } else {
        console.error('Failed to fetch images:', data.message);
      }
    } catch (error) {
      console.error('Error fetching images:', error);
    }
  };

  const handleViewPatientProfile = (patientId) => {
    console.log(`Viewing patient profile for patientId: ${patientId}`);
  };

  const handleCloseModal = () => {
    setOpenModal(false);
  };

  const handleUserClick = (userId) => {
    fetchWorkContents(userId);
  };

  return (
    <div className="min-h-screen flex flex-col items-center bg-indigo-700 text-white">
      <div className="container mx-auto mt-8">
        <div className="flex flex-col items-center justify-center">
          <h1 className="text-3xl font-bold mb-4">Find Patient</h1>
          <div className="mb-4">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={handleSearch}
              disabled={loading}
              placeholder="Patient ID"
              className="px-4 py-2 border rounded text-black"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {searchResults.map((user) => (
            <div
              key={user.id}
              className={`p-4 border rounded shadow hover:bg-gray-100 cursor-pointer ${cardStatus[user.id] === 'error' ? 'bg-red-200 cursor-not-allowed' : ''}`}
              onClick={() => cardStatus[user.id] !== 'error' && handleUserClick(user.id)}
              disabled={cardStatus[user.id] === 'error'}
            >
              <p className="font-bold text-lg">{user.fName} {user.lName}</p>
              <p className="text-white-600">ID: <span className="text-white-600">{user.id}</span></p>
            </div>
          ))}
        </div>
        <div className="overflow-x-auto w-full">
          <table className="min-w-full bg-blue-700 border border-gray-200">
            <thead>
              <tr className="bg-blue-800 border-b border-gray-200">
                <th className="px-6 py-3 text-left text-xs font-semibold text-white uppercase tracking-wider">File Name</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-white uppercase tracking-wider">Creation Date</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-white uppercase tracking-wider">Finished Date</th>
                <th className="px-6 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {workContents.map(workContent => (
                <tr key={workContent.id} className="bg-blue-700">
                  <td className="px-6 py-4 whitespace-nowrap text-white">{workContent.fileName}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-white">{moment(workContent.createdAt).format('YYYY-MM-DD HH:mm:ss')}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-white">{workContent.finishedAt ? moment(workContent.finishedAt).format('YYYY-MM-DD HH:mm:ss') : 'N/A'}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <button className="text-blue-300 hover:text-blue-500 font-semibold mr-4" onClick={() => handleViewResult(workContent.id)}>View Result</button>
                    <button className="text-blue-300 hover:text-blue-500 font-semibold" onClick={() => handleViewPatientProfile(workContent.patientId)}>View Patient Profile</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <Modal open={openModal} onClose={handleCloseModal} images={modalImages} />
      <ToastContainer />
    </div>
  );
}
