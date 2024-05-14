// UserContext.js

import { createContext, useState } from 'react';

export const UserContext = createContext();

export const UserProvider = ({ children }) => {
  const [fName, setFName] = useState(null);

  return (
    <UserContext.Provider value={{ fName, setFName }}>
      {children}
    </UserContext.Provider>
  );
};
