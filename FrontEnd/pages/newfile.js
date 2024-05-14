import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { uploadFile } from '../services/index'; // Import the uploadFile function
import Cookies from 'js-cookie';

export default function NewFile() {
  const [fileType, setFileType] = useState('mri');
  const [duration, setDuration] = useState(1);
  const [selectedFile, setSelectedFile] = useState(null);
//   const [userId, setUserId] = useState(null);
  const router = useRouter();
  const userID = Cookies.get('userId');
  const handleSubmit = async (e) => {
    e.preventDefault();



    try {
      // Call the uploadFile function with selectedFile and fileType
      const res = await uploadFile(selectedFile, fileType,userID);
      if (res.success) {
        console.log('File uploaded successfully');
        router.push('/home');
      } else {
        console.error('Failed to upload file');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    <div className="w-full h-screen flex items-center justify-center bg-gray-200">
      <div className="bg-white p-8 rounded-lg shadow-md">
        <h2 className="text-2xl font-bold mb-4">Upload New File</h2>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="fileType" className="block font-semibold mb-2">File Type</label>
            <div>
              <label className="inline-flex items-center">
                <input
                  type="radio"
                  name="fileType"
                  value="mri"
                  checked={fileType === 'mri'}
                  onChange={() => setFileType('mri')}
                  className="mr-2"
                />
                MRI
              </label>
              <label className="inline-flex items-center ml-4">
                <input
                  type="radio"
                  name="fileType"
                  value="ct"
                  checked={fileType === 'ct'}
                  onChange={() => setFileType('ct')}
                  className="mr-2"
                />
                CT
              </label>
            </div>
          </div>
          <div className="mb-4">
            <label htmlFor="duration" className="block font-semibold mb-2">Duration</label>
            <select
              id="duration"
              value={duration}
              onChange={(e) => setDuration(e.target.value)}
              className="border p-2 w-full"
            >
              {[...Array(12).keys()].map((value) => (
                <option key={value} value={value + 1}>{value + 1}</option>
              ))}
            </select>
          </div>
          <div className="mb-4">
            <label htmlFor="file" className="block font-semibold mb-2">Browse File</label>
            <input
              type="file"
              id="file"
              onChange={(e) => setSelectedFile(e.target.files[0])}
              className="border p-2 w-full"
            />
          </div>
          <button type="submit" className="bg-blue-500 text-white font-semibold py-2 px-4 rounded hover:bg-blue-600">
            Upload
          </button>
        </form>
      </div>
    </div>
  );
}
