import React, { useEffect, useContext } from 'react';
import Cookies from 'js-cookie';
import Router from 'next/router';
import { UserContext } from '../services/UserContext';

export default function Home() {
  const token = Cookies.get('token');
  const { fName, setFName, userId } = useContext(UserContext);

  useEffect(() => {
    if (!token) {
      Router.push('/');
    } else {
      const storedFName = Cookies.get('fName');
      // const storedUserId = Cookies.get('userId');
      if (storedFName) {
        setFName(storedFName);
      }
      // if (storedUserId) {
      //   setUserId(storedUserId);
      // }
    }
  }, []);

  const logout = () => {
    Cookies.remove('token');
    Cookies.remove('fName'); // Remove fName cookie
    Cookies.remove('userId'); // Remove userId cookie
    Router.push('/');
  };

  const handleNewFile = () => {
    Router.push('/newfile');
  };

  const handleViewWork = () => {
    Router.push('/userdashboard');
  };

  return (
    <div className='w-full h-screen flex items-center justify-center text-white bg-indigo-700 flex-col tracking-widest uppercase'>
      {fName && (
        <p className='text-4xl font-extrabold mb-4'>Welcome {fName} to the home Page</p>
      )}
      <div className='flex space-x-4'>
        <button onClick={handleNewFile} className='bg-white border-2 border-white hover:bg-transparent transition-all text-indigo-700 hover:text-white font-semibold text-lg  px-4 py-2 rounded duration-700'>
          New File
        </button>
        <button onClick={handleViewWork} className='bg-white border-2 border-white hover:bg-transparent transition-all text-indigo-700 hover:text-white font-semibold text-lg  px-4 py-2 rounded duration-700'>
          View Work
        </button>
      </div>
      <button onClick={logout} className='bg-white border-2 border-white hover:bg-transparent transition-all text-indigo-700 hover:text-white font-semibold text-lg  px-4 py-2 rounded duration-700 mt-4'>
        Logout
      </button>
    </div>
  );
}
