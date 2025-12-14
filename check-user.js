const { MongoClient } = require('mongodb');

async function checkUser() {
  const uri = 'mongodb://localhost:27017/';
  const client = new MongoClient(uri);

  try {
    await client.connect();
    const db = client.db('nda-company');
    
    const users = await db.collection('User').find({}).toArray();
    
    console.log(`\n📊 Total users in database: ${users.length}\n`);
    
    if (users.length > 0) {
      users.forEach((user, index) => {
        console.log(`User ${index + 1}:`);
        console.log(`  Email: ${user.email}`);
        console.log(`  Name: ${user.fullName}`);
        console.log(`  Company: ${user.company}`);
        console.log(`  Role: ${user.role}`);
        console.log(`  Created: ${user.createdAt}`);
        console.log('');
      });
    } else {
      console.log('❌ No users found in database');
    }
    
  } catch (error) {
    console.error('Error:', error.message);
  } finally {
    await client.close();
  }
}

checkUser();

