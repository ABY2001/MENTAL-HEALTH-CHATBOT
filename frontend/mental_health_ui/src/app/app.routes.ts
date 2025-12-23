import { Routes } from '@angular/router';
import { MainLayout } from './core/layout/main-layout/main-layout';

export const routes: Routes = [
    {
        path:'login',
        loadChildren: ()=> import('./features/login/login.routing').then(m =>m.LoginRoutes)
        
    },
    {
      path: '',redirectTo: '/login',pathMatch: 'full'
    },
    
    {
        path: '',
        component: MainLayout,
        children:[
            {
                path: '',
                loadChildren: ()=> import('./features/audio-record/audio-record').then(m =>m.AudioRecord)
            }
        ]
    }  
];
